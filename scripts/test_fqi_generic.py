import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from deep_ifs.models.generic_policy import GenericPolicy
import argparse
import atexit
import joblib
import numpy as np
import pandas as pd
from deep_ifs.envs.atari import Atari
from deep_ifs.evaluation.evaluation import evaluate_policy
from deep_ifs.extraction.Autoencoder import Autoencoder
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import LinearRegression
from deep_ifs.models.mlp import MLP
from deep_ifs.extraction.GenericEncoder import GenericEncoder
from deep_ifs.extraction.NNStack import NNStack

parser = argparse.ArgumentParser()
parser.add_argument('load-fq', type=str, default=None)
parser.add_argument('--episodes', type=int, default=10)
parser.add_argument('--ae-id', type=str, default='AE_SS')
parser.add_argument('--nn-stack', action='store_true')
args = parser.parse_args()

initial_actions = [1, 4, 5]  # Initial actions for BreakoutDeterministic-v3

mdp = Atari('BreakoutDeterministic-v3', clip_reward=True)


if args.nn_stack:
    # Use features from AE + NN0
    fe = NNStack()
    # Reward network
    nn = GenericEncoder('/home/tesla/Projects/nips2017-deepIFS/output/NN0_reward_feature/nn_stack_0/encoder_0.h5', binarize=False)
    support = np.load('/home/tesla/Projects/nips2017-deepIFS/output/NN0_reward_feature/nn_stack_0/support_0.npy')
    fe.add(nn, support)
    # Autoencoder
    ae = Autoencoder((4, 108, 84),
                     nb_epochs=300,
                     encoding_dim=512,
                     binarize=True,
                     logger=None,
                     ckpt_file='autoencoder_ckpt.h5')
    ae.load('/home/tesla/Projects/nips2017-deepIFS/output/%s/autoencoder_ckpt.h5' % args.ae_id)
    support = np.array([True] * 640)
    fe.add(ae, support)
else:
    # Use features from autoencoder
    fe = Autoencoder((4, 108, 84),
                     nb_epochs=300,
                     encoding_dim=512,
                     binarize=True,
                     logger=None,
                     ckpt_file='autoencoder_ckpt.h5')
    fe.load('/home/tesla/Projects/nips2017-deepIFS/output/%s/autoencoder_ckpt.h5' % args.ae_id)
    support = np.array([True] * 640)
    fe.set_support(support)
    
print 'Loading data'
F, Q = joblib.load(args.load_fq)

print 'Buiding model'
m = ExtraTreesRegressor(n_estimators=50, min_samples_split=5, min_samples_leaf=2, n_jobs=-1)
# m = LinearRegression(n_jobs=-1)
# m = MLP(F.shape[1], 6, layers=(512, 512))

print 'Fitting model'
# Fit model
if isinstance(m, MLP):
    m.fit(F, Q, epochs=20, patience=0, validation_data=(F_test, Q_test))
else:
    m.fit(F, Q)
    
print 'Building policy'
policy = GenericPolicy(fe, m, [0, 1, 2, 3, 4, 5], epsilon=0.05)

print 'Evaluating policy'
es_evaluation = evaluate_policy(mdp,
                                policy,
                                n_episodes=args.episodes,
                                initial_actions=initial_actions)
                                
print 'Results: %s' % str(es_evaluation)
