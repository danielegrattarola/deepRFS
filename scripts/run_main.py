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
# TODO debug
debug = False
sars_episodes = 100
# TODO debug
nn_nb_epochs = 2 if debug else 50
alg_iterations = 100  # Number of algorithm steps to make
# TODO debug
rec_steps = 1 if debug else 100  # Number of recursive steps to make
ifs_nb_trees = 50  # Number of trees to use in IFS
ifs_significance = 0.3  # Significance for IFS
fqi_iterations = 100  # Number of steps to train FQI
r2_change_threshold = 1.05  # Threshold for IFS confidence below which to stop algorithm
# END ARGS

# ADDITIONAL OBJECTS
logger = Logger(output_folder='../output/')
evaluation_results = []
# END ADDITIONAL OBJECTS

nn_stack = NNStack()  # To store all neural networks and IFS supports

mdp = Atari('BreakoutDeterministic-v3')
action_values = mdp.action_space.values
nb_actions = mdp.action_space.n

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
policy = EpsilonFQI(fqi_params, nn_stack, epsilon=1.0)  # Do not unpack the dict
toc()

for i in range(alg_iterations):
    tic('Collecting SARS dataset')
    sars = collect_sars(mdp, policy, nn_stack, episodes=sars_episodes, debug=debug)  # State, action, reward, next_state
    # sars_class_weight = get_class_weight(sars)
    sars_sample_weight = get_sample_weight(sars)
    toc()

    tic('Fitting NN0')
    target_size = 1  # Initial target is the scalar reward
    nn = ConvNet(mdp.state_shape, target_size, nb_actions=nb_actions,
                 sample_weight=sars_sample_weight,
                 nb_epochs=nn_nb_epochs)  # Maps frames to reward
    nn.fit(pds_to_npa(sars.S), pds_to_npa(sars.A), pds_to_npa(sars.R))
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
    # If scikit-learn version is < 0.19 this will throw a warning,
    # keep it like this anyway
    ifs_y = ifs_y.reshape(-1, 1)
    ifs.fit(ifs_x, ifs_y)
    support = ifs.get_support()
    nb_new_features = np.array(support).sum()
    r2_change = (ifs.scores_[-1] - ifs.scores_[0]) / abs(ifs.scores_[0])
    toc('IFS - New features: %s; R2 change: %s' % (nb_new_features, r2_change))

    # TODO Debug
    if debug:
        support[2] = True
    nn_stack.add(nn, support)

    for j in range(1, rec_steps + 1):
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
        nn = ConvNet(image_shape, target_size, nb_actions=nb_actions,
        			 nb_epochs=nn_nb_epochs)  # Maps frames to residual support dynamics
        nn.fit(pds_to_npa(sares.S), pds_to_npa(sares.A), pds_to_npa(sares.RES).squeeze())
        toc()

        tic('Building FADF dataset for IFS')
        fadf = build_fadf(nn_stack, nn, sars, sfadf)  # All features, action, dynamics, all next_features
        toc()

        tic('Running IFS with target D')
        ifs = IFS(**ifs_params)
        ifs_x, ifs_y = split_dataset_for_ifs(fadf, features='F', target='D')
        preload_features = range(nn_stack.get_support_dim())
        ifs.fit(ifs_x, ifs_y, preload_features=preload_features)
        support = ifs.get_support()
        support = support[len(preload_features):]  # Remove already selected features from support
        nb_new_features = np.array(support).sum()
        r2_change = (ifs.scores_[-1] - ifs.scores_[0]) / abs(ifs.scores_[0])
        toc('IFS - New features: %s; R2 change: %s' % (nb_new_features, r2_change))

        nn_stack.add(nn, support)

        if nb_new_features == 0 or r2_change < r2_change_threshold:
            print 'Done.'
            print '#' * 50 + '\n'
            break

    tic('Building global FARF dataset for FQI')
    global_farf = build_global_farf(nn_stack, sars)  # All features, action, reward, all next_features
    toc()

    tic('Updating policy')
    sast, r = split_dataset_for_fqi(global_farf)
    all_features_dim = nn_stack.get_support_dim()  # Need to pass new dimension of "states" to instantiate new FQI
    # Need to update ActionRegressor to only use the actions actually in the dataset
    regressor = Regressor(regressor_class=ExtraTreesRegressor,
                          **fqi_regressor_params)
    action_values = pds_to_npa(global_farf.A.unique())
    regressor = ActionRegressor(regressor,
                                discrete_actions=action_values,
                                tol=0.5,
                                **fqi_regressor_params)
    policy.fqi_params['estimator'] = regressor
    # TODO if it crashes, update policy.nn_stack here
    policy.fit_on_dataset(sast, r, all_features_dim)
    policy.epsilon_step()
    toc()

    tic('Evaluating policy after update')
    evaluation_metrics = evaluate_policy(mdp, policy, nn_stack)
    evaluation_results.append(evaluation_metrics)
    toc(evaluation_results)

# Plot evaluation results
evaluation_results = pd.DataFrame(evaluation_results,
                                  columns=['Score', 'Confidence (score)', 'Steps', 'Confidence (steps)'])
evaluation_results[['Score', 'Steps']].plot().get_figure().savefig(logger.path + 'evaluation.png')

# TODO Log run configuration



