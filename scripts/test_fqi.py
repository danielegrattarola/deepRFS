import argparse
import joblib
from deep_ifs.envs.atari import Atari
from deep_ifs.evaluation.evaluation import evaluate_policy
from deep_ifs.extraction.NNStack import NNStack
from deep_ifs.models.epsilonFQI import EpsilonFQI
from deep_ifs.utils.Logger import Logger
from deep_ifs.utils.timer import tic, toc

parser = argparse.ArgumentParser()
parser.add_argument('fqi-model', type=str, default=None)
parser.add_argument('nn-stack', type=str, default=None)
parser.add_argument('-d', '--debug', action='store_true')
parser.add_argument('-e', '--env', type=str, default=None)
parser.add_argument('--episodes', type=int, default=10)
parser.add_argument('--save-video', action='store_true')
args = parser.parse_args()

max_eval_steps = 2 if args.debug else 500  # Max length of evaluation episodes

logger = Logger(output_folder='../output/')
mdp = Atari('BreakoutDeterministic-v3')

tic('Reading data...')
nn_stack = NNStack()
nn_stack.load(args.nn_stack)
policy = EpsilonFQI(None, nn_stack, fqi=joblib.load(args.fqi_model))
toc()

tic('Evaluation (%s episodes)' % args.episodes)
evaluation_metrics = evaluate_policy(mdp,
                                     policy,
                                     max_ep_len=max_eval_steps,
                                     n_episodes=args.episodes,
                                     save_video=args.save_video,
                                     save_path=logger.path)

toc('Done: %s' % (evaluation_metrics))