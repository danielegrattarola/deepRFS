import argparse
import numpy as np
import joblib
from deep_ifs.envs.atari import Atari
from deep_ifs.evaluation.evaluation import evaluate_policy
from deep_ifs.extraction.Autoencoder import Autoencoder
from deep_ifs.models.epsilonFQI import EpsilonFQI
from deep_ifs.utils.Logger import Logger
from deep_ifs.utils.timer import tic, toc, log

parser = argparse.ArgumentParser()
parser.add_argument('fqi_model', type=str, default=None, help='Path to a saved FQI pickle to load as policy')
parser.add_argument('fe', type=str, default=None, help='Path to a saved NNStack folder to load as feature extractor')
parser.add_argument('-d', '--debug', action='store_true', help='Run in debug mode')
parser.add_argument('--save-video', action='store_true', help='Save the gifs of the evaluation episodes')
parser.add_argument('-e', '--env', type=str, default='BreakoutDeterministic-v4', help='Atari environment on which to run the algorithm')
parser.add_argument('--episodes', type=int, default=10, help='Number of episodes to run in evaluation')
args = parser.parse_args()

logger = Logger(output_folder='../output/', custom_run_name='test_fqi%Y%m%d-%H%M%S')
mdp = Atari(args.env)

# Feature extraction
fe = Autoencoder((4, 108, 84),
                 nb_epochs=300,
                 binarize=args.binarize,
                 logger=logger,
                 ckpt_file='autoencoder_ckpt.h5')
fe.load(args.fe)
# Set support for feature extractor
support = joblib.load(args.support)
fe.set_support(support)

# Policy
policy = EpsilonFQI(None, fe, fqi=args.fqi_model)

log('Using:\n'
    '\tfqi-model: %s\n'
    '\tnn_stack: %s' % (args.fqi_model, args.nn_stack))

tic('Evaluation (%s episodes)' % args.episodes)
evaluation_metrics = evaluate_policy(mdp,
                                     policy,
                                     n_episodes=args.episodes,
                                     save_video=args.save_video,
                                     save_path=logger.path)

toc('Done: %s' % str(evaluation_metrics))