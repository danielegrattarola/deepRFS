# TODO Documentation
import matplotlib

# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
import argparse
from deep_ifs.envs.atari import Atari
import joblib
from deep_ifs.extraction.NNStack import NNStack
from deep_ifs.extraction.ConvNet import ConvNet
from deep_ifs.extraction.ConvNetClassifier import ConvNetClassifier
from deep_ifs.models.epsilonFQI import EpsilonFQI
from deep_ifs.utils.datasets import *
from deep_ifs.utils.Logger import Logger
from deep_ifs.utils.timer import tic, toc, log
from ifqi.models import Regressor, ActionRegressor
from matplotlib import pyplot as plt
from sklearn.ensemble import ExtraTreesRegressor


# ARGS
parser = argparse.ArgumentParser()
parser.add_argument('path', type=str, help='Path to the NN h5 file')
parser.add_argument('sars', type=str, help='Path to the sars dataset pickle')
parser.add_argument('-d', '--debug', action='store_true',
                    help='Run in debug mode')
parser.add_argument('--save-video', action='store_true',
                    help='Save the gifs of the evaluation episodes')
parser.add_argument('-e', '--env', type=str, default='BreakoutDeterministic-v3',
                    help='Atari environment on which to run the algorithm')
parser.add_argument('--binarize', action='store_true',
                    help='Binarize input to the neural networks')
parser.add_argument('--classify', action='store_true',
                    help='Use a classifier for NN0')
args = parser.parse_args()
# END ARGS

tic('Setup')
# HYPERPARAMETERS
sars_episodes = 10 if args.debug else 50  # Number of SARS episodes to collect
nn_nb_epochs = 2 if args.debug else 300  # Number of training epochs for NNs

# SETUP
logger = Logger(output_folder='../output/', custom_run_name='test%Y%m%d-%H%M%S')
nn_stack = NNStack()  # To store all neural networks and FS supports
mdp = Atari(args.env, clip_reward=args.classify)
action_values = mdp.action_space.values
nb_actions = mdp.action_space.n

# Create policy
fqi_regressor_params = {'n_estimators': 50,
                        'n_jobs': -1}
regressor = ActionRegressor(Regressor(regressor_class=ExtraTreesRegressor,
                                      **fqi_regressor_params),
                            discrete_actions=action_values,
                            tol=0.5)
fqi_params = {'estimator': regressor,
              'state_dim': nn_stack.get_support_dim(),
              'action_dim': 1,  # Action is discrete monodimensional
              'discrete_actions': action_values,
              'verbose': True}
policy = EpsilonFQI(fqi_params, nn_stack)  # Do not unpack the dict
toc()

tic('Loading data')
# 4 frames, action, reward, 4 frames
sars = joblib.load(args.sars)
sars_sample_weight = get_sample_weight(sars)

S = pds_to_npa(sars.S)  # 4 frames
A = pds_to_npa(sars.A)  # Discrete action
R = pds_to_npa(sars.R)  # Scalar reward
toc()

tic('Building model')
# NN maps frames to reward
if args.classify:
    from sklearn.preprocessing import OneHotEncoder

    ohe = OneHotEncoder(sparse=False)
    R = ohe.fit_transform(R.reshape(-1, 1) - R.min())
    nb_classes = R.shape[1]  # Target is the one-hot encoded reward
    nn = ConvNetClassifier(mdp.state_shape,
                           nb_classes,
                           nb_actions=nb_actions,
                           l1_alpha=0.0,
                           sample_weight=sars_sample_weight,
                           nb_epochs=nn_nb_epochs,
                           binarize=args.binarize)
else:
    target_size = 1  # Initial target is the scalar reward
    nn = ConvNet(mdp.state_shape,
                 target_size,
                 nb_actions=nb_actions,
                 l1_alpha=0.01,
                 sample_weight=sars_sample_weight,
                 nb_epochs=nn_nb_epochs,
                 binarize=args.binarize)

nn.load(args.path)
toc()

tic('Predicting')
pred = nn.predict(S, A)
# idxs = np.argmax(pred, 1)
# oh_pred = np.zeros(pred.shape)
# oh_pred[np.arange(len(idxs)), idxs] = 1
toc()

tic('Plotting')
for i in range(R.shape[1]):
    plt.figure()
    plt.scatter(R[:, i].reshape(-1), pred[:, i].reshape(-1))
    plt.savefig(logger.path + 'NN0_test_%s_v_reward.png' % i)
    plt.close()
toc()

log('Done')
