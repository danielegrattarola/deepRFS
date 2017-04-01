import argparse
import joblib
from deep_ifs.envs.atari import Atari
from deep_ifs.evaluation.evaluation import evaluate_policy
from deep_ifs.extraction.NNStack import NNStack
from deep_ifs.models.epsilonFQI import EpsilonFQI
from deep_ifs.utils.Logger import Logger
from deep_ifs.utils.timer import tic, toc, log

parser = argparse.ArgumentParser()
parser.add_argument('fqi_model', type=str, default=None)
parser.add_argument('nn_stack', type=str, default=None)
parser.add_argument('-d', '--debug', action='store_true')
parser.add_argument('-e', '--env', type=str, default=None)
parser.add_argument('--episodes', type=int, default=10)
parser.add_argument('--save-video', action='store_true')
args = parser.parse_args()

max_eval_steps = 2 if args.debug else 500  # Max length of evaluation episodes

logger = Logger(output_folder='../output/', custom_run_name='test%Y%m%d-%H%M%S')
mdp = Atari('BreakoutDeterministic-v3')

tic('Reading data...')
nn_stack = NNStack()
nn_stack.load(args.nn_stack)
policy = EpsilonFQI(None, nn_stack, fqi=joblib.load(args.fqi_model))
toc()

log('Using:\n'
    '\tnn_stack: %s\n'
    '\tfqi: %s' % (args.nn_stack, args.fqi_model))

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