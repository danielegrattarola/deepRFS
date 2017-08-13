import argparse
from deep_ifs.envs.atari import Atari
from deep_ifs.models.epsilonFQI import EpsilonFQI
from deep_ifs.utils.datasets import *
from deep_ifs.extraction.ConvNet import ConvNet
from deep_ifs.extraction.NNStack import NNStack
from deep_ifs.utils.helpers import *
from matplotlib import pyplot as plt

# Read arguments
parser = argparse.ArgumentParser()
parser.add_argument('--dataset-dir', type=str, default='NN.h5',
                    help='Path to the NN h5 file')
args = parser.parse_args()

mdp = Atari('BreakoutDeterministic-v4')
action_values = mdp.action_space.values
nb_actions = mdp.action_space.n

nn_stack = NNStack()  # To store all neural networks and IFS supports
fqi_params = {'estimator': None,
              'state_dim': 10,  # Don't care
              'action_dim': 1,  # Action is discrete monodimensional
              'discrete_actions': action_values,
              'gamma': mdp.gamma,
              'horizon': 1,
              'verbose': True}
policy = EpsilonFQI(fqi_params, nn_stack, epsilon=1.0)  # Do not unpack the dict

sars_episodes = 30
sars = collect_sars(mdp, policy, episodes=sars_episodes)  # State, action, reward, next_state
sars_sample_weight = get_sample_weight(sars)

target_size = 1  # Target is the scalar reward
nn = ConvNet(mdp.state_shape, target_size, nb_actions=nb_actions,
             sample_weight=sars_sample_weight,
             nb_epochs=0)  # Maps frames to reward

nn.load(args.dataset_dir)
s = pds_to_npa(sars.S)
a = pds_to_npa(sars.A)
r = pds_to_npa(sars.R)

nn.fit(s, a, r)
r_hat = nn.predict(s, a)

plt.figure()
plt.subplot(2, 1, 1)
plt.plot(r)
plt.subplot(2, 1, 2)
plt.plot(r_hat)

plt.figure()
plt.scatter(r, r_hat)

plt.show()
