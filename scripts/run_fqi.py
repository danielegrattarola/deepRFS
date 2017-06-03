# TODO Documentation
import matplotlib

from deep_ifs.utils.datasets import build_faft_r_from_disk

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
from deep_ifs.utils.Logger import Logger
from deep_ifs.utils.timer import tic, toc, log
from ifqi.models import Regressor, ActionRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neural_network import MLPRegressor
from tqdm import tqdm
from xgboost import XGBRegressor


def exit_callback():
    log('\n\nIf you want to test a policy use the following:')
    log('\tFQI model: %s' % logger.path + 'fqi_step_X_eval_Y.pkl')
    log('\tNNStack: %s' % args.base_folder + 'nn_stack_%s/' % args.iteration_id)
atexit.register(exit_callback)

# ARGS
parser = argparse.ArgumentParser()
parser.add_argument('nn_stack', type=str,
                    help='Path to nn_stack folder')
parser.add_argument('sars', type=str,
                    help='Path to sars folder')
parser.add_argument('iteration_id', type=int,
                    help='Index of run_main step saved in the base folder that '
                         'you want to use')
parser.add_argument('-d', '--debug', action='store_true',
                    help='Run in debug mode')
parser.add_argument('--save-video', action='store_true',
                    help='Save the gifs of the evaluation episodes')
parser.add_argument('-e', '--env', type=str, default='BreakoutDeterministic-v3',
                    help='Atari environment on which to run the algorithm')
parser.add_argument('--iter', type=int, default=100,
                    help='Number of fqi iterations to run')
parser.add_argument('--episodes', type=int, default=10,
                    help='Number of episodes to run at each evaluation step')
parser.add_argument('--eval-freq', type=int, default=5,
                    help='Period (number of steps) with which to run evaluation'
                         ' steps')
parser.add_argument('--fqi-model-type', type=str, default='extra',
                    help='Type of model to use for fqi (\'linear\', \'ridge\', '
                         '\'extra\', \'xgb\', \'mlp\')')
parser.add_argument('--clip', action='store_true',
                    help='Clip reward')
args = parser.parse_args()

max_eval_steps = 2 if args.debug else 1000  # Max length of evaluation episodes
initial_actions = [1, 4, 5]  # Initial actions for BreakoutDeterministic-v3

# SETUP
log('Reading data')
# Load NNStack
nn_stack = NNStack()  # To store all neural networks and IFS support
nn_stack.load(args.base_folder + 'nn_stack_%s/' % args.iteration_id)

# Load dataset for FQI
faft, r, action_values = build_faft_r_from_disk(nn_stack, args.sars)

if args.clip:
    r = np.clip(r, -1, 1)

# if args.use_sw and args.fqi_model_type != 'xgb':
#     sample_weight = get_sample_weight(r,
#                                       balanced=True,
#                                       round_target=True)
# else:
#     sample_weight = None

log('Got %s samples' % len(faft))

log('Setup')
logger = Logger(output_folder='../output/', custom_run_name='fqi%Y%m%d-%H%M%S')
evaluation_results = []
mdp = Atari(args.env)
nb_actions = mdp.action_space.n

log('Creating policy')
# Create policy
# Create ActionRegressor
if args.fqi_model_type == 'extra':
    fqi_regressor_params = {'n_estimators': 50,
                            'max_depth': 1 + nn_stack.get_support_dim() / 2,
                            'n_jobs': -1}
    fqi_regressor_class = ExtraTreesRegressor
elif args.fqi_model_type == 'xgb':
    fqi_regressor_params = {'max_depth': 8,
                            'n_estimators': 100}
    fqi_regressor_class = XGBRegressor
elif args.fqi_model_type == 'linear':
    fqi_regressor_params = {'n_jobs': -1}
    fqi_regressor_class = LinearRegression
elif args.fqi_model_type == 'ridge':
    fqi_regressor_params = {}
    fqi_regressor_class = Ridge
elif args.fqi_model_type == 'mlp':
    fqi_regressor_params = {'hidden_layer_sizes': (128, 128),
                            'early_stopping': True}
    fqi_regressor_class = MLPRegressor
else:
    raise NotImplementedError('Allowed models: \'extra\', \'linear\', '
                              '\'ridge\', \'xgb\', \'mlp\'.')

regressor = ActionRegressor(Regressor(regressor_class=fqi_regressor_class,
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

n_reward_features = nn_stack.get_support_dim(index=0)
log('\n%s reward features' % n_reward_features)
log('%s dynamics features\n' % (nn_stack.get_support_dim() - n_reward_features))

# Initial fit
policy.partial_fit(faft, r)
for i in tqdm(range(args.iter)):
    policy.partial_fit()
    if i % args.eval_freq == 0 or i == (args.iter-1):
        evaluation_metrics = evaluate_policy(mdp,
                                             policy,
                                             n_episodes=args.episodes,
                                             save_video=args.save_video,
                                             save_path=logger.path,
                                             append_filename='fqi_iter_%03d' % i,
                                             initial_actions=initial_actions)
        evaluation_results.append(evaluation_metrics)
        # Save fqi policy
        policy.save_fqi(logger.path + 'fqi_iter_%03d_score_%s.pkl' %
                        (i, int(evaluation_results[-1][0])))
        tqdm.write('Iter %s: %s' % (i, evaluation_results[-1]))

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

