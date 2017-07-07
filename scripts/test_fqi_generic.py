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
initial_actions = [1, 4, 5]  # Initial actions for BreakoutDeterministic-v3

mdp = Atari('BreakoutDeterministic-v3', clip_reward=True)

# Use features from autoencoder
ae = Autoencoder((4, 108, 84),
                 nb_epochs=300,
                 encoding_dim=512,
                 binarize=True,
                 logger=None,
                 ckpt_file='autoencoder_ckpt.h5')
ae.load('/home/tesla/Projects/nips2017-deepIFS/output/ae_rfs20170615-173517/autoencoder_ckpt.h5')

F, Q = joblib.load('')
m = ExtraTreesRegressor(n_estimators=50, min_samples_split=5, min_samples_leaf=2, n_jobs=-1)

m.fit(F, Q)

policy = GenericPolicy(ae, m, [0, 1, 2, 3, 4, 5], epsilon=0.05)
es_evaluation = evaluate_policy(mdp,
                                policy,
                                n_episodes=10,
                                initial_actions=initial_actions)