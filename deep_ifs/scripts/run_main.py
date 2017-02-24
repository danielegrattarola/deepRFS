"""
Algorithm pseudocode

NN[i]: is the i-th neural network and provides the features(S) method that
	returns all 512 features produced when state S is given as input to the network
  and the the s_features(S) method that returns all **selected** features
  produced when state S is given as input to each network.

NN_stack: contains all trained neural networks so far, provides the s_features(S)
	method that returns all **selected** features produced when state S is given as
	input to each network.

Policy = fully random

Main loop:

    Collect SARS' samples with policy
    Fix dataset to account for imbalance (proportional to the number of transitions for each reward class)

    Fit neural network NN[0]: S -> R, using SARS' dataset

    Build FARF' dataset using SARS' dataset:
        F = NN[0].features(S)
        A = A
        R = R
        F' = NN[0].features(S')
    Select support features of NN[0] with IFS using FARF' dataset (target = R)

    For i in range(1, N):
        Build SFADF' dataset using SARS' dataset:
            S = S
            F = NN_stack.s_features(S)
            A = A
            D = NN[i-1].s_features(S) - NN[i-1].s_features(S')
            F' = NN_stack.s_features(S')

        Fit model M: F -> D, using SFADF' dataset

        Build SARes dataset from SFADF':
            S = S
            A = A
            Res = D - M(F)
        Fit neural network NNi: S -> Res, using SARes dataset

        Build new FADF' dataset from SARS' and SFADF':
            F = NN_stack.s_features(S) + NN[i].features(S)
            A = A
            D = SFADF'.D
            F' = NN_stack.s_features(S') + NN[i].features(S')
        Select support features of NNi with IFS using new FADF' dataset

        If (no new feature is selected) or (R2 of D is below a threshold):
            Break

    Update policy with FQI (using support features of all steps), decrease randomicity
"""

# TODO Documentation for all classes/methods

from ifqi.models import Regressor, ActionRegressor
from deep_ifs.models.epsilonFQI import EpsilonFQI
from deep_ifs.evaluation.evaluation import *
from deep_ifs.extraction.NNStack import NNStack
from deep_ifs.extraction.ConvNet import ConvNet
from deep_ifs.selection.ifs import IFS
from deep_ifs.utils.datasets import *
from deep_ifs.envs.atari import Atari
from sklearn.ensemble import ExtraTreesRegressor

# ARGS
alg_iterations = 100  # Number of algorithm steps to make
rec_steps = 100  # Number of recursive steps to make
fqi_iterations = 100  # Number of steps to train FQI
# END ARGS

nn_stack = NNStack()  # To store all neural networks and IFS supports

mdp = Atari()
action_values = mdp.action_space.values

# Create epsilon FQI model
# Action regressor of ExtraTreesRegressor for FQI
fqi_regressor_params = {'n_estimators': 50,
                        'criterion': 'mse',
                        'min_samples_split': 5,
                        'min_samples_leaf': 2,
                        'input_scaled': False,
                        'output_scaled': False,
                        'n_jobs': -1}
regressor = Regressor(regressor_class=ExtraTreesRegressor,
                      **fqi_regressor_params)
regressor = ActionRegressor(regressor,
                            discrete_actions=action_values,
                            tol=0.5,
                            **fqi_regressor_params)
fqi_params = {'estimator': regressor,
              'state_dim': 10,  # Don't care at this step
              'action_dim': 1,  # Action is discrete monodimensional
              'discrete_actions': action_values,
              'gamma': mdp.gamma,
              'horizon': fqi_iterations,
              'verbose': True}
policy = EpsilonFQI(fqi_params, epsilon=1.0)  # Do not unpack the dict

for i in range(alg_iterations):
    sars = collect_sars(mdp, policy)  # State, action, reward, next_state
    sars_class_weight = get_class_weights(sars)

    target_size = 1  # Initial target is the scalar reward
    nn = ConvNet(mdp.state_shape, target_size, class_weight=sars_class_weight)  # Maps frames to reward
    nn.fit(sars.s, sars.r)

    farf = build_farf(nn, sars)  # Features, action, reward, next_features

    # TODO Parameters for IFS
    ifs_params = {}
    ifs = IFS(**ifs_params)
    ifs_x, ifs_y = split_dataset_for_ifs(farf, features='F', target='R')
    ifs.fit(ifs_x, ifs_y)
    support = ifs.get_support()

    nn_stack.add(nn, support)

    for j in range(1, rec_steps):
        prev_support_dim = nn_stack.get_support_dim()

        # State, all features, action, support dynamics, all next_features
        sfadf = build_sfadf(nn_stack, nn, support,sars)

        # TODO Parameters for ExtraTreeRegressor
        # TODO Should this be a neural network, too?
        model = ExtraTreesRegressor()
        model.fit(sfadf.f, sfadf.d)

        sares = build_sares(model, sfadf)  # Res = D - model(F)

        # TODO Do we need to convert the class weights to sample weights to give the same importance to samples as in the reward case?
        image_shape = sares.S.head(1)[0].shape
        target_size = sares.RES.head(1)[0].shape[0]  # Target is the residual support dynamics
        nn = ConvNet(image_shape, target_size)  # Maps frames to residual support dynamics
        nn.fit(sares.S, sares.RES)

        fadf = build_fadf(nn_stack, nn, sars, sfadf)  # All features, action, dynamics, all next_features

        # TODO Parameters for IFS
        ifs_params = {}
        ifs = IFS(**ifs_params)
        ifs_x, ifs_y = split_dataset_for_ifs(fadf, features='F', target='D')
        # TODO Preload features for IFS
        ifs.fit(ifs_x, ifs_y, preload_features=preload_features)

        # TODO Don't add the support like this because IFS will also return all the previously selected features
        support = ifs.get_support()
        nn_stack.add(nn, support)

        # TODO Confidence threshold
        if nn_stack.get_support_dim() <= prev_support_dim or ifs.scores_confidences_ < threshold:
            print 'Done.'
            break

    global_farf = build_global_farf(nn_stack, sars)  # All features, action, reward, all next_features

    sast, r = split_dataset_for_fqi(global_farf)
    all_features_dim = nn_stack.get_support_dim()  # Need to pass new dimension of "states" to instantiate new FQI
    policy.fit_on_dataset(sast, r, all_features_dim)
    policy.epsilon_step()

    evaluate_policy(mdp, policy, nn_stack)
    # TODO plot/save evaluation results