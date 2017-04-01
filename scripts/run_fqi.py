# TODO Documentation
import matplotlib
matplotlib.use('Agg')  # Force matplotlib to not use any Xwindows backend.
import argparse
import atexit
import joblib
import numpy as np
import pandas as pd
from deep_ifs.envs.atari import Atari
from deep_ifs.evaluation.evaluation import evaluate_policy
from deep_ifs.extraction.NNStack import NNStack
from deep_ifs.models.epsilonFQI import EpsilonFQI
from deep_ifs.utils.datasets import get_sample_weight, pds_to_npa, split_dataset_for_fqi
from deep_ifs.utils.Logger import Logger
from deep_ifs.utils.timer import tic, toc, log
from ifqi.models import Regressor, ActionRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import LinearRegression, Ridge
from tqdm import tqdm


def exit_callback():
    log('\n\nIf you want to test a policy use the following:\n'
        '\tfqi-model: %s\n'
        '\tnn-stack: %s' % (logger.path + 'fqi_step_X_eval_Y.pkl',
                          args.base_folder + 'nn_stack_%s/' % args.iteration_id))
atexit.register(exit_callback)

# ARGS
parser = argparse.ArgumentParser()
parser.add_argument('base_folder', type=str, help='Path to run folder with global_farf dataset and nn_stack_X/ folder')
parser.add_argument('iteration_id', type=int, help='Index of run_main step saved in the base folder that you want to use')
parser.add_argument('-d', '--debug', action='store_true', help='Run in debug mode')
parser.add_argument('--save-video', action='store_true', help='Save the gifs of the evaluation episodes')
parser.add_argument('-e', '--env', type=str, default='BreakoutDeterministic-v3', help='Atari environment on which to run the algorithm')
parser.add_argument('--iter', type=int, default=100, help='Number of fqi iterations to run')
parser.add_argument('--episodes', type=int, default=10, help='Number of episodes to run at each evaluation step')
parser.add_argument('--eval-freq', type=int, default=5, help='Period (number of steps) with which to run evaluation steps')
parser.add_argument('--fqi-model-type', type=str, default='extra', help='Type of model to use for fqi (\'linear\', \'ridge\', \'extra\')')
parser.add_argument('--sample-weights', action='store_true', help='Use sample weights to train FQI')
args = parser.parse_args()

max_eval_steps = 2 if args.debug else 500  # Max length of evaluation episodes

# SETUP
tic('Reading data...')
nn_stack = NNStack()  # To store all neural networks and IFS support
nn_stack.load(args.base_folder + 'nn_stack_%s/' % args.iteration_id)
global_farf = pd.read_pickle(args.base_folder + 'global_farf_%s.pickle' % args.iteration_id)
farf_sample_weight = get_sample_weight(global_farf) if args.sample_weights else None
toc()

tic('Setup...')
logger = Logger(output_folder='../output/', custom_run_name='fqi%Y%m%d-%H%M%S')
evaluation_results = []
mdp = Atari(args.env)
nb_actions = mdp.action_space.n

tic('Building dataset for FQI')
sast, r = split_dataset_for_fqi(global_farf)
all_features_dim = nn_stack.get_support_dim()  # Need to pass new dimension of "states" to instantiate new FQI
action_values = np.unique(pds_to_npa(global_farf.A))
toc()

tic('Creating policy')
# Create policy
if args.fqi_model_type == 'extra':
    fqi_regressor_params = {'n_estimators': 50,
                            'n_jobs': -1}
    regressor = ActionRegressor(Regressor(regressor_class=ExtraTreesRegressor,
                                          **fqi_regressor_params),
                                discrete_actions=action_values,
                                tol=0.5)
elif args.fqi_model_type == 'linear':
    fqi_regressor_params = {'n_jobs': -1}
    regressor = ActionRegressor(Regressor(regressor_class=LinearRegression,
                                          **fqi_regressor_params),
                                discrete_actions=action_values,
                                tol=0.5)
elif args.fqi_model_type == 'ridge':
    fqi_regressor_params = {'n_jobs': -1}
    regressor = ActionRegressor(Regressor(regressor_class=Ridge,
                                          **fqi_regressor_params),
                                discrete_actions=action_values,
                                tol=0.5)

state_dim = nn_stack.get_support_dim()
fqi_params = {'estimator': regressor,
              'state_dim': state_dim,
              'action_dim': 1,  # Action is discrete monodimensional
              'discrete_actions': action_values,
              'gamma': mdp.gamma,
              'horizon': args.iter,
              'verbose': False}
policy = EpsilonFQI(fqi_params, nn_stack)  # Do not unpack the dict
toc()
toc()

nb_reward_features = nn_stack.get_support_dim(index=0)
log('\n%s reward features' % nb_reward_features)
log('%s dynamics features\n' % (nn_stack.get_support_dim() - nb_reward_features))

# Initial fit
policy.partial_fit_on_dataset(sast, r, sample_weight=farf_sample_weight)
for i in tqdm(range(args.iter)):
    policy.partial_fit_on_dataset(sample_weight=farf_sample_weight)
    if i % args.eval_freq == 0 or i == (args.iter-1):
        tqdm.write('Step %s: started eval...' % i)
        evaluation_metrics = evaluate_policy(mdp,
                                             policy,
                                             max_ep_len=max_eval_steps,
                                             n_episodes=args.episodes,
                                             save_video=args.save_video,
                                             save_path=logger.path,
                                             append_filename='step_%s' % i)
        evaluation_results.append(evaluation_metrics)
        # Save fqi policy
        joblib.dump(policy.fqi, logger.path + 'fqi_step_%s_eval_%s.pkl' %
                    (i, int(evaluation_results[-1][0])))
        tqdm.write('Step %s: %s' % (i, evaluation_results[-1]))

# FINAL OUTPUT #
# Plot evaluation results
evaluation_results = pd.DataFrame(evaluation_results,
                                  columns=['score', 'confidence_score',
                                           'steps', 'confidence_steps'])
evaluation_results.to_csv('evaluation.csv', index=False)
fig = evaluation_results['score'].plot().get_figure()
fig.savefig(logger.path + 'evaluation_score.png')
fig = evaluation_results['steps'].plot().get_figure()
fig.savefig(logger.path + 'evaluation_steps.png')

