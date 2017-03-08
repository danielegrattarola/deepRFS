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

        If (no new feature is selected) or (R2 of added features is below a threshold):
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
from deep_ifs.utils.Logger import Logger
from deep_ifs.utils.timer import *
from deep_ifs.envs.atari import Atari
from sklearn.ensemble import ExtraTreesRegressor

tic('Initial setup')
# ARGS
sars_episodes = 100
nn_nb_epochs = 30
alg_iterations = 100  # Number of algorithm steps to make
rec_steps = 100  # Number of recursive steps to make
ifs_nb_trees = 50  # Number of trees to use in IFS
ifs_significance = 0.1  # Significance for IFS
fqi_iterations = 100  # Number of steps to train FQI
confidence_threshold = 0.01  # Threshold for IFS confidence below which to stop algorithm
# END ARGS

# ADDITIONAL OBJECTS
logger = Logger(output_folder='../output/')
evaluation_results = []
# END ADDITIONAL OBJECTS

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
toc()

for i in range(alg_iterations):
    tic('Collecting SARS dataset')
    sars = collect_sars(mdp, policy, episodes=sars_episodes)  # State, action, reward, next_state
    sars_class_weight = get_class_weight(sars)
    toc()

    tic('Fitting NN0')
    target_size = 1  # Initial target is the scalar reward
    nn = ConvNet(mdp.state_shape, target_size, class_weight=sars_class_weight,
                 nb_epochs=nn_nb_epochs)  # Maps frames to reward
    nn.fit(pds_to_npa(sars.S), pds_to_npa(sars.R))
    toc()

    tic('Building FARF dataset for IFS')
    farf = build_farf(nn, sars)  # Features, action, reward, next_features
    toc()

    tic('Running IFS with target R')
    ifs_estimator_params = {'n_estimators': ifs_nb_trees,
                            'n_jobs': -1}
    ifs_params = {'estimator': ExtraTreesRegressor(**ifs_estimator_params),
                  'n_features_step': 1,
                  'cv': None,
                  'scale': True,
                  'verbose': 1,
                  'significance': ifs_significance}
    ifs = IFS(**ifs_params)
    ifs_x, ifs_y = split_dataset_for_ifs(farf, features='F', target='R')
    ifs_y = ifs_y.reshape(-1, 1)
    ifs.fit(ifs_x, ifs_y)
    support = ifs.get_support()
    toc()

    nn_stack.add(nn, support)

    for j in range(1, rec_steps):
        prev_support_dim = nn_stack.get_support_dim()

        tic('Building SFADF dataset for residuals model')
        # State, all features, action, support dynamics, all next_features
        sfadf = build_sfadf(nn_stack, nn, support, sars)
        toc()

        tic('Fitting residuals model')
        model = ExtraTreesRegressor(n_estimators=50)  # This should slightly underfit
        model.fit(pds_to_npa(sfadf.F), pds_to_npa(sfadf.D))
        toc()

        tic('Building SARes dataset')
        sares = build_sares(model, sfadf)  # Res = D - model(F)
        toc()

        tic('Fitting NN%s' % j)
        image_shape = sares.S.head(1)[0].shape
        if sares.RES.head(1)[0].shape != (1,):
            target_size = sares.RES.head(1)[0].squeeze().shape[0]  # Target is the residual support dynamics
        else:
            target_size = sares.RES.head(1)[0].shape[0]
        nn = ConvNet(image_shape, target_size)  # Maps frames to residual support dynamics
        nn.fit(pds_to_npa(sares.S), pds_to_npa(sares.RES).squeeze())
        toc()

        tic('Building FADF dataset for IFS')
        fadf = build_fadf(nn_stack, nn, sars, sfadf)  # All features, action, dynamics, all next_features
        toc()

        tic('Running IFS with target D')
        ifs = IFS(**ifs_params)
        ifs_x, ifs_y = split_dataset_for_ifs(fadf, features='F', target='D')
        preload_features = range(nn_stack.get_support_dim())
        ifs.fit(ifs_x, ifs_y, preload_features=preload_features)
        # TODO Ask Pirotta: preload_features are returned at the beginning of the support?
        support = ifs.get_support()
        support = support[len(preload_features):]  # Remove already selected features from support
        toc()

        # print '# new features', np.array(support).sum()
        # print 'Len preload features', len(preload_features)
        # print 'Full IFS support', ifs.get_support()
        # print 'Support', support
        # raw_input()

        nn_stack.add(nn, support)

        # TODO Ask Pirotta: how to implement confidence threshold
        if np.array(support).sum() == 0:
            print 'Done.'
            break

    tic('Building global FARF dataset for FQI')
    global_farf = build_global_farf(nn_stack, sars)  # All features, action, reward, all next_features
    toc()

    tic('Updating policy')
    sast, r = split_dataset_for_fqi(global_farf)
    all_features_dim = nn_stack.get_support_dim()  # Need to pass new dimension of "states" to instantiate new FQI
    policy.fit_on_dataset(sast, r, all_features_dim)
    policy.epsilon_step()
    toc()

    tic('Evaluating policy after update')
    evaluation_metrics = evaluate_policy(mdp, policy, nn_stack)
    evaluation_results.append(evaluation_metrics)
    toc()

# Plot evaluation results
evaluation_results = pd.DataFrame(evaluation_results)
evaluation_results.plot().get_figure().savefig(logger.path + 'evaluation.png')

# TODO Log run configuration



