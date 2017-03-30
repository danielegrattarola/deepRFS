# TODO Documentation
import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
import pandas as pd
import numpy as np
import argparse
from tqdm import tqdm
from ifqi.models import Regressor, ActionRegressor
from deep_ifs.models.epsilonFQI import EpsilonFQI
from deep_ifs.evaluation.evaluation import evaluate_policy
from deep_ifs.extraction.NNStack import NNStack
from deep_ifs.utils.datasets import pds_to_npa, split_dataset_for_fqi
from deep_ifs.utils.Logger import Logger
from deep_ifs.utils.timer import tic, toc, log
from deep_ifs.envs.atari import Atari
from sklearn.linear_model import Ridge
from deep_ifs.utils.datasets import get_sample_weight

# ARGS
parser = argparse.ArgumentParser()
parser.add_argument('base_folder', type=str, help='path to run folder with dataset and nn_stack')
parser.add_argument('iteration_id', type=int, help='number of iteration saved in the base folder that you want to use')
parser.add_argument('-d', action='store_true', help='debug')
parser.add_argument('--save-video', action='store_true', help='save evaluation gifs')
parser.add_argument('--iter', type=int, default=100, help='fqi iterations')
parser.add_argument('--episodes', type=int, default=10, help='number of evaluation episodes to run at each step')
args = parser.parse_args()

max_eval_steps = 2 if args.d else 4000  # Maximum length of evaluation episodes

# SETUP
# Read from disk
tic('Reading data...')
nn_stack = NNStack()  # To store all neural networks and IFS support
nn_stack.load(args.base_folder + 'nn_stack_%s/' % args.iteration_id)
global_farf = pd.read_pickle(args.base_folder + 'global_farf_%s.pickle' % args.iteration_id)
farf_sample_weight = get_sample_weight(global_farf)
toc()

tic('Setup...')
logger = Logger(output_folder='../output/')
evaluation_results = []
mdp = Atari('BreakoutDeterministic-v3')
nb_actions = mdp.action_space.n

tic('Building dataset for FQI')
sast, r = split_dataset_for_fqi(global_farf)
all_features_dim = nn_stack.get_support_dim()  # Need to pass new dimension of "states" to instantiate new FQI
action_values = np.unique(pds_to_npa(global_farf.A))
toc()

# Create policy
fqi_regressor_params = {}
regressor = Regressor(regressor_class=Ridge)
regressor = ActionRegressor(regressor,
                            discrete_actions=action_values,
                            tol=0.5,
                            **fqi_regressor_params)
state_dim = nn_stack.get_support_dim()
fqi_params = {'estimator': regressor,
              'state_dim': state_dim,  # Don't care at this step
              'action_dim': 1,  # Action is discrete monodimensional
              'discrete_actions': action_values,
              'gamma': mdp.gamma,
              'horizon': args.iter,
              'verbose': True}
policy = EpsilonFQI(fqi_params, nn_stack)  # Do not unpack the dict
toc()

nb_reward_features = nn_stack.get_support_dim(index=0)
log('%s reward features' % nb_reward_features)
log('%s dynamics features' % (nn_stack.get_support_dim() - nb_reward_features))

# Initial fit
policy.partial_fit_on_dataset(sast, r, sample_weight=farf_sample_weight)
for i in tqdm(range(args.iter)):
    policy.partial_fit_on_dataset(sample_weight=farf_sample_weight)
    evaluation_metrics = evaluate_policy(mdp,
                                         policy,
                                         max_ep_len=max_eval_steps,
                                         n_episodes=args.episodes,
                                         save_video=args.save_video,
                                         save_path=logger.path)
    evaluation_results.append(evaluation_metrics)
    tqdm.write('Step %s: %s', (i, evaluation_results[-1][[0, 2]]))

# FINAL OUTPUT #
# Plot evaluation results
evaluation_results = pd.DataFrame(evaluation_results,
                                  columns=['score', 'confidence_score',
                                           'steps', 'confidence_steps'])
evaluation_results.to_csv('evaluation.csv', index=False)
fig = evaluation_results[['score', 'steps']].plot().get_figure()
fig.savefig(logger.path + 'evaluation.png')

