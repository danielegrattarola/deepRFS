import argparse
import numpy as np
from deep_ifs.envs.atari import Atari
from deep_ifs.evaluation.evaluation import evaluate_policy
from deep_ifs.extraction.NNStack import NNStack
from deep_ifs.models.epsilonFQI import EpsilonFQI
from deep_ifs.utils.Logger import Logger
from deep_ifs.utils.timer import tic, toc, log

parser = argparse.ArgumentParser()
parser.add_argument('fqi_model', type=str, default=None,
                    help='Path to a saved FQI pickle to load as policy')
parser.add_argument('nn_stack', type=str, default=None,
                    help='Path to a saved NNStack folder to load as feature '
                         'extractor')
parser.add_argument('-d', '--debug', action='store_true',
                    help='Run in debug mode')
parser.add_argument('--save-video', action='store_true',
                    help='Save the gifs of the evaluation episodes')
parser.add_argument('-e', '--env', type=str, default='BreakoutDeterministic-v3',
                    help='Atari environment on which to run the algorithm')
parser.add_argument('--episodes', type=int, default=10,
                    help='Number of episodes to run in evaluation')
parser.add_argument('--max-eval-steps', type=int, default=500,
                    help='Max number of steps in an episode (-1 for inf)')
args = parser.parse_args()

# Max length of evaluation episodes
if args.debug:
    max_eval_steps = 2
elif args.max_eval_steps == -1:
    max_eval_steps = np.inf
else:
    max_eval_steps = args.max_eval_steps

logger = Logger(output_folder='../output/', custom_run_name='test%Y%m%d-%H%M%S')
mdp = Atari(args.env)

tic('Reading data')
nn_stack = NNStack()
nn_stack.load(args.nn_stack)
policy = EpsilonFQI(None, nn_stack, fqi=args.fqi_model)
toc()

log('Using:\n'
    '\tfqi-model: %s\n'
    '\tnn_stack: %s' % (args.fqi_model, args.nn_stack))

nb_reward_features = nn_stack.get_support_dim(index=0)
log('\n%s reward features' % nb_reward_features)
log('%s dynamics features\n' % (nn_stack.get_support_dim() - nb_reward_features))

tic('Evaluation (%s episodes)' % args.episodes)
evaluation_metrics = evaluate_policy(mdp,
                                     policy,
                                     max_ep_len=max_eval_steps,
                                     n_episodes=args.episodes,
                                     save_video=args.save_video,
                                     save_path=logger.path)

toc('Done: %s' % str(evaluation_metrics))