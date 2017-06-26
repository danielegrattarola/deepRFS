import matplotlib
matplotlib.use('Agg')  # Force matplotlib to not use any Xwindows backend.
import argparse
import joblib
import numpy as np
import pandas as pd
from deep_ifs.envs.atari import Atari
from deep_ifs.evaluation.evaluation import evaluate_policy
from deep_ifs.extraction.Autoencoder import Autoencoder
from deep_ifs.models.epsilonFQI import EpsilonFQI
from deep_ifs.utils.datasets import build_faft_r_from_disk
from deep_ifs.utils.Logger import Logger
from deep_ifs.utils.timer import tic, toc, log, setup_logging
from ifqi.models import Regressor, ActionRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neural_network import MLPRegressor
from tqdm import tqdm
from xgboost import XGBRegressor


# Args
parser = argparse.ArgumentParser()
parser.add_argument('ae', type=str,
                    help='Path to ae')
parser.add_argument('support', type=str,
                    help='Path to support')
parser.add_argument('sars', type=str,
                    help='Path to sars folder')
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
parser.add_argument('--binarize', action='store_true',
                    help='Binarize input to the neural networks')
args = parser.parse_args()

# Params
max_eval_steps = 2 if args.debug else 1000  # Max length of evaluation episodes
initial_actions = [1, 4, 5]  # Initial actions for BreakoutDeterministic-v3

# Setup
logger = Logger(output_folder='../output/', custom_run_name='fqi%Y%m%d-%H%M%S')
setup_logging(logger.path + 'log.txt')

# Feature extraction
target_size = 1
ae = Autoencoder((4, 108, 84),
                 nb_epochs=300,
                 encoding_dim=512,
                 binarize=args.binarize,
                 logger=logger,
                 ckpt_file='autoencoder_ckpt.h5')
ae.load(args.ae)

# Set support for AE
support = joblib.load(args.support)
ae.set_support(support)

# Load dataset for FQI
log('Building dataset for FQI')
faft, r, action_values = build_faft_r_from_disk(ae, args.sars)
if args.clip:
    r = np.clip(r, -1, 1)
log('Got %s samples' % len(faft))

# Environment
mdp = Atari(args.env, clip_reward=args.clip)
nb_actions = mdp.action_space.n

log('Creating policy')
# Create policy
# Create ActionRegressor
if args.fqi_model_type == 'extra':
    fqi_regressor_params = {'n_estimators': 50,
                            'min_samples_split': 5,
                            'min_samples_leaf': 2,
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

state_dim = len(support)
fqi_params = {'estimator': regressor,
              'state_dim': state_dim,
              'action_dim': 1,  # Action is discrete monodimensional
              'discrete_actions': action_values,
              'gamma': mdp.gamma,
              'horizon': args.iter,
              'verbose': False}
policy = EpsilonFQI(fqi_params, ae)  # Do not unpack the dict

# Fit FQI
log('Fitting FQI')
evaluation_results = []
fqi_patience = args.iter
fqi_current_patience = fqi_patience
fqi_best = (-np.inf, 0, -np.inf, 0)

policy.partial_fit(faft, r)
for partial_iter in tqdm(range(args.iter)):
    policy.partial_fit()
    if partial_iter % args.eval_freq == 0 or partial_iter == (args.iter-1):
        es_evaluation = evaluate_policy(mdp,
                                        policy,
                                        n_episodes=args.episodes,
                                        initial_actions=initial_actions,
                                        save_video=args.save_video,
                                        save_path=logger.path,
                                        append_filename='fqi_iter_%03d' % partial_iter)
        evaluation_results.append(es_evaluation)
        tqdm.write('Iter %s: %s' % (partial_iter, evaluation_results[-1]))
        # Save fqi policy
        if es_evaluation[0] > fqi_best[0]:
            tqdm.write('Saving best policy')
            fqi_best = es_evaluation
            fqi_current_patience = fqi_patience
            policy.save_fqi(logger.path + 'fqi_iter_%03d_score_%s.pkl' %
                            (partial_iter, int(evaluation_results[-1][0])))
        else:
            fqi_current_patience -= 1
            if fqi_current_patience == 0:
                break

# Final output
evaluation_results = pd.DataFrame(evaluation_results,
                                  columns=['score', 'confidence_score',
                                           'steps', 'confidence_steps'])
evaluation_results.to_csv('evaluation.csv', index=False)
fig = evaluation_results['score'].plot().get_figure()
fig.savefig(logger.path + 'evaluation_score.png')
fig = evaluation_results['steps'].plot().get_figure()
fig.savefig(logger.path + 'evaluation_steps.png')

