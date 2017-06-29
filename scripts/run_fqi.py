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
from deep_ifs.utils.timer import log, setup_logging
from ifqi.models import Regressor, ActionRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neural_network import MLPRegressor
from tqdm import tqdm
from xgboost import XGBRegressor

parser = argparse.ArgumentParser()

envarg = parser.add_argument_group('Environment')
envarg.add_argument("--screen_width", type=int, default=84, help="Screen width after resize.")
envarg.add_argument("--screen_height", type=int, default=110, help="Screen height after resize.")

memarg = parser.add_argument_group('Replay memory')
memarg.add_argument("--replay_size", type=int, default=1000000, help="Maximum size of replay memory.")
memarg.add_argument("--history_length", type=int, default=4, help="How many screen frames form a state.")

netarg = parser.add_argument_group('Deep Q-learning network')
netarg.add_argument("--learning_rate", type=float, default=0.00025, help="Learning rate.")
netarg.add_argument("--discount_rate", type=float, default=0.99, help="Discount rate for future rewards.")
netarg.add_argument("--batch_size", type=int, default=32, help="Batch size for neural network.")
netarg.add_argument('--optimizer', choices=['rmsprop', 'adam', 'adadelta'], default='rmsprop', help='Network optimization algorithm.')
netarg.add_argument("--decay_rate", type=float, default=0.95, help="Decay rate for RMSProp and Adadelta algorithms.")
netarg.add_argument("--clip_error", type=float, default=1, help="Clip error term in update between this number and its negative.")
netarg.add_argument("--target_steps", type=int, default=10000, help="Copy main network to target network after this many steps.")
netarg.add_argument("--min_reward", type=float, default=-1, help="Minimum reward.")
netarg.add_argument("--max_reward", type=float, default=1, help="Maximum reward.")
netarg.add_argument("--batch_norm", type=bool, default=False, help="Use batch normalization in all layers.")

neonarg = parser.add_argument_group('Neon')
neonarg.add_argument('--backend', choices=['cpu', 'gpu'], default='gpu', help='backend type')
neonarg.add_argument('--device_id', type=int, default=0, help='gpu device id (only used with GPU backend)')
neonarg.add_argument('--datatype', choices=['float16', 'float32', 'float64'], default='float32', help='default floating point precision for backend [f64 for cpu only]')
neonarg.add_argument('--stochastic_round', const=True, type=int, nargs='?', default=False, help='use stochastic rounding [will round to BITS number of bits if specified]')

antarg = parser.add_argument_group('Agent')
antarg.add_argument("--exploration_rate_start", type=float, default=1, help="Exploration rate at the beginning of decay.")
antarg.add_argument("--exploration_rate_end", type=float, default=0.1, help="Exploration rate at the end of decay.")
antarg.add_argument("--exploration_decay_steps", type=float, default=1000000, help="How many steps to decay the exploration rate.")
antarg.add_argument("--exploration_rate_test", type=float, default=0.05, help="Exploration rate used during testing.")
antarg.add_argument("--train_frequency", type=int, default=4, help="Perform training after this many game steps.")
antarg.add_argument("--train_repeat", type=int, default=1, help="Number of times to sample minibatch during training.")
antarg.add_argument("--random_starts", type=int, default=30, help="Perform max this number of dummy actions after game restart, to produce more random game dynamics.")

mainarg = parser.add_argument_group('Main loop')
mainarg.add_argument("--random_steps", type=int, default=50000, help="Populate replay memory with random steps before starting learning.")
mainarg.add_argument("--train_steps", type=int, default=250000, help="How many training steps per epoch.")
mainarg.add_argument("--test_steps", type=int, default=125000, help="How many testing steps after each epoch.")
mainarg.add_argument("--epochs", type=int, default=200, help="How many epochs to run.")
mainarg.add_argument("--start_epoch", type=int, default=0, help="Start from this epoch, affects exploration rate and names of saved snapshots.")
mainarg.add_argument("--load_weights", type=str, help="Load network from file.")
mainarg.add_argument("--save_weights_prefix", help="Save network to given file. Epoch and extension will be appended.")

comarg = parser.add_argument_group('Common')
comarg.add_argument("--num_episodes", type=int, default=100, help="Number of episodes to test.")
comarg.add_argument("--random_seed", type=int, help="Random seed for repeatable experiments.")
comarg.add_argument("--num_blocks", type=int, default=100, help="Number of episodes to test.")


parser.add_argument('model', type=str,
                    help='Path to feature extractor')
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
parser.add_argument('--faft', type=str,
                    help='Load FAFT, R and action values for FQI from file')
parser.add_argument('--use-dqn', action='store_true',
                    help='Use DQN instead of AE for feature extraction')
args = parser.parse_args()

# Params
max_eval_steps = 2 if args.debug else 1000  # Max length of evaluation episodes
initial_actions = [1, 4, 5]  # Initial actions for BreakoutDeterministic-v3

# Setup
logger = Logger(output_folder='../output/', custom_run_name='fqi%Y%m%d-%H%M%S')
setup_logging(logger.path + 'log.txt')

# Environment
mdp = Atari(args.env, clip_reward=args.clip)
nb_actions = mdp.action_space.n

# Feature extraction
if args.use_dqn:
    from deep_ifs.extraction.DeepQNetwork import DeepQNetwork
    fe = DeepQNetwork(nb_actions, args)
    fe.load_weights(args.model)
else:
    target_size = 1
    fe = Autoencoder((4, 108, 84),
                     nb_epochs=300,
                     encoding_dim=512,
                     binarize=args.binarize,
                     logger=logger,
                     ckpt_file='autoencoder_ckpt.h5')
    fe.load(args.model)

# Set support for feature extractor
support = joblib.load(args.support)
fe.set_support(support)

# Load dataset for FQI
log('Building dataset for FQI')
if args.faft is not None:
    faft, r, action_values = joblib.load(args.faft)
else:
    faft, r, action_values = build_faft_r_from_disk(fe, args.sars)
if args.clip:
    r = np.clip(r, -1, 1)
log('Got %s samples' % len(faft))

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
policy = EpsilonFQI(fqi_params, fe)  # Do not unpack the dict

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

