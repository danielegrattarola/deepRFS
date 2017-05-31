import argparse
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats.kde import gaussian_kde
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import ExtraTreesRegressor

from deep_ifs.extraction.ConvNet import ConvNet
from deep_ifs.extraction.GenericEncoder import GenericEncoder
from deep_ifs.utils.datasets import pds_to_npa

parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, default='../output/run_ifs20170529-113019/', help='Run folder')
args = parser.parse_args()

sars = np.load('../data/BreakoutDeterministic-v3/sars_0.npy')


idx_0 = np.argwhere(sars[:, 2] == 0).ravel()
idx_1 = np.argwhere(sars[:, 2] == 1).ravel()
idx_1 = np.random.choice(idx_1, 10)
idx = np.append(idx_0, idx_1)
np.random.shuffle(idx)

sars = sars[idx]

en = GenericEncoder(args.path + 'NN0_encoder_step0.h5')
s = np.load(args.path + 'nn_stack_0/support_0.npy')

f = en.s_features(pds_to_npa(sars[:, 0]), s)
ff = en.s_features(pds_to_npa(sars[:, 3]), s)
dyn = f - ff

models = [LinearRegression, ExtraTreesRegressor]
for model in models:
    if isinstance(model, ExtraTreesRegressor):
        m = model(min_samples_split=5, min_samples_leaf=2)
    else:
        m = model()
    m.fit(f, dyn)
    p = m.predict(f)
    res = p.ravel() - dyn.ravel()
    pdf = gaussian_kde(res.T)
    w = 1. / pdf(res.T)

    plt.figure()
    ax = plt.subplot(6, 1, 1)
    ax.title.set_text('F')
    plt.plot(f)

    ax = plt.subplot(6, 1, 2)
    ax.title.set_text('FF')
    plt.plot(ff)

    ax = plt.subplot(6, 1, 3)
    ax.title.set_text('Dyn')
    plt.plot(dyn)

    ax = plt.subplot(6, 1, 4)
    ax.title.set_text('Predictions')
    plt.plot(p)

    ax = plt.subplot(6, 1, 5)
    ax.title.set_text('Res')
    plt.plot(res)

    ax = plt.subplot(6, 1, 6)
    ax.title.set_text('W')
    plt.plot(w)

plt.show()
