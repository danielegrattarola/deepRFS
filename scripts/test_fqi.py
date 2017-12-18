import argparse

import joblib
import numpy as np

from deep_rfs.envs.atari import Atari
from deep_rfs.evaluation.evaluation import evaluate_policy
from deep_rfs.extraction.Autoencoder import Autoencoder
from deep_rfs.models.epsilonFQI import EpsilonFQI
from deep_rfs.utils.Logger import Logger
from deep_rfs.utils.timer import tic, toc

# Args
# Main
parser = argparse.ArgumentParser()
parser.add_argument('-d', '--debug', action='store_true', help='Run in debug mode')

# MDP
parser.add_argument('-e', '--env', type=str, default='BreakoutDeterministic-v4', help='Atari environment on which to run the algorithm')
parser.add_argument('--clip', action='store_true', help='Clip reward of MDP')
parser.add_argument('--clip-eval', action='store_true', help='Clip reward of MDP during evaluation')

# AE
parser.add_argument('--load-ae', type=str, default=None, help='Path to h5 weights file to load into AE')
parser.add_argument('--load-ae-support', type=str, default=None, help='Path to file with AE support')
parser.add_argument('--binarize', action='store_true', help='Binarize input to the neural networks')
parser.add_argument('--use-sw', action='store_true', help='Use sample weights when training AE')
parser.add_argument('--use-vae', action='store_true', help='Use VAE instead of usual AE')
parser.add_argument('--vae-beta', type=float, default=1., help='Beta hyperparameter for Beta-VAE')
parser.add_argument('--use-dense', action='store_true', help='Use AE with dense inner layer instead of usual AE')
parser.add_argument('--dropout', type=float, default=0., help='Dropout rate for dense AE')
parser.add_argument('--n-features', type=int, default=128, help='Number of features for contractive, dense and VAE')

# FQI
parser.add_argument('--load-fqi', type=str, default=None, help='Path to fqi file to load into policy')
parser.add_argument('--save-video', action='store_true', help='Save the gifs of the evaluation episodes')
parser.add_argument('--eval-episodes', type=int, default=10, help='Number of testing episodes to run')
args = parser.parse_args()

# Env
initial_actions = [1]  # Initial actions for BreakoutDeterministic-v4

# AE
nn_nb_epochs = 5 if args.debug else 100  # Number of training epochs for NNs
nn_batch_size = 6 if args.debug else 32  # Number of samples in a batch for AE (len(sars) will be multiple of this number)
nn_binarization_threshold = 0.35 if args.env == 'PongDeterministic-v4' else 0.1

logger = Logger(output_folder='../output/', custom_run_name='test_fqi%Y%m%d-%H%M%S')

# Environment
mdp = Atari(args.env, clip_reward=args.clip)
action_values = mdp.action_space.values

# Autoencoder (this one will be used as FE, but never trained)
ae = Autoencoder((4, 108, 84),
                 n_features=args.n_features,
                 batch_size=nn_batch_size,
                 nb_epochs=nn_nb_epochs,
                 binarize=args.binarize,
                 binarization_threshold=nn_binarization_threshold,
                 logger=logger,
                 ckpt_file='autoencoder_ckpt.h5',
                 use_vae=args.use_vae,
                 beta=args.vae_beta,
                 use_dense=args.use_dense,
                 dropout_prob=args.dropout)
ae.model.summary()
ae.load(args.load_ae)
if args.load_ae_support is not None:
    support = joblib.load(args.load_ae_support)  # Load support from previous training
else:
    support = np.array([True] * ae.get_features_number())  # Keep all features
ae.set_support(support)

# Policy
fqi_params = args.load_fqi
policy = EpsilonFQI(fqi_params, ae)

tic('Evaluation (%s episodes)' % args.eval_episodes)
# Evaluate policy after loading
partial_eval = evaluate_policy(mdp,
                               policy,
                               n_episodes=args.eval_episodes,
                               initial_actions=initial_actions,
                               save_video=args.save_video,
                               save_path=logger.path,
                               append_filename='fqi_test_after_loading',
                               eval_epsilon=0.05,
                               clip=args.clip_eval)

toc('Done: %s' % str(partial_eval))