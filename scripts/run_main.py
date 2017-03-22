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

# TODO Documentation
import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
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
from matplotlib import pyplot as plt

tic('Initial setup')
# ARGS
# TODO debug
debug = False
farf_analysis = True
r2_analysis = False
# TODO debug
sars_episodes = 10 if debug else 200  # Number of SARS episodes to collect
# TODO debug
nn_nb_epochs = 2 if debug else 300  # Number of epochs for the networks
alg_iterations = 100  # Number of steps to make in the main loop
# TODO debug
rec_steps = 1 if debug else 100  # Number of recursive steps to make
ifs_nb_trees = 50  # Number of trees to use in IFS
ifs_significance = 0.01  # Significance for IFS
# TODO debug
fqi_iterations = 2 if debug else 100  # Number of steps to train FQI
r2_change_threshold = 0.10  # % of IFS improvement below which to stop loop
# TODO debug
max_eval_steps = 2 if debug else 1000  # Maximum length of evaluation episodes
# END ARGS

# ADDITIONAL OBJECTS
logger = Logger(output_folder='../output/')
evaluation_results = []

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
# END ADDITIONAL OBJECTS
toc()

for i in range(alg_iterations):
    # NEURAL NETWORK 0 #
    tic('Collecting SARS dataset')
    # 4 frames, action, reward, 4 frames
    sars = collect_sars(mdp, policy, episodes=sars_episodes, debug=debug)
    sars_sample_weight = get_sample_weight(sars)
    S = pds_to_npa(sars.S)  # 4 frames
    A = pds_to_npa(sars.A)  # Discrete action
    R = pds_to_npa(sars.R)  # Scalar reward
    toc('Got %s SARS\' samples' % len(sars))

    tic('Resetting NN stack')
    nn_stack.reset()  # Clear the stack after collecting sars' with last policy
    log('Policy stack outputs %s features' % policy.nn_stack.get_support_dim())
    toc()

    tic('Fitting NN0')
    target_size = 1  # Initial target is the scalar reward
    # NN maps frames to reward
    nn = ConvNet(mdp.state_shape, target_size, nb_actions=nb_actions,
                 l1_alpha=0.01, sample_weight=sars_sample_weight,
                 nb_epochs=nn_nb_epochs)
    nn.fit(S, A, R)
    nn.load('NN.h5')  # Load best network (saved by callback)

    log('Cleaning memory (S, A, R)')
    del S, A, R
    toc()
    # END NEURAL NETWORK 0 #

    # ITERATIVE FEATURE SELECTION 0 #
    tic('Building FARF dataset for IFS')
    farf = build_farf(nn, sars)  # Features, action, reward, next_features
    ifs_x, ifs_y = split_dataset_for_ifs(farf, features='F', target='R')
    # If scikit-learn version is < 0.19 this will throw a warning
    ifs_y = ifs_y.reshape(-1, 1)
    # Print the number of nonzero features
    nonzero_mfv_counts = np.count_nonzero(np.mean(ifs_x[:-1], axis=0))
    log('Number of non-zero feature: %s' % nonzero_mfv_counts)
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
    ifs.fit(ifs_x, ifs_y, preload_features=(range(511) if r2_analysis else None))  # TODO R2 analysis
    support = ifs.get_support()
    got_action = support[-1]
    support = support[:-1]  # Remove action from support
    nb_new_features = np.array(support).sum()
    r2_change = (ifs.scores_[-1] - ifs.scores_[0]) / abs(ifs.scores_[0])
    log('IFS - New features: %s' % nb_new_features)
    log('Action was%s selected' % ('' if got_action else ' NOT'))
    log('R2 change %s (from %s to %s)' %
        (r2_change, ifs.scores_[0], ifs.scores_[-1]))

    if not farf_analysis:  # TODO farf analysis
        log('Cleaning memory (farf, ifs_x, ifs_y)')
        del farf, ifs_x, ifs_y
    toc()
    # END ITERATIVE FEATURE SELECTION 0 #

    # TODO Debug
    if debug:
        support[2] = True

    # TODO farf analysis
    if farf_analysis:
        feature_idxs = np.argwhere(support).reshape(-1)
        print 'Unique targets', np.unique(ifs_y)
        for f in feature_idxs:
            plt.figure()
            plt.scatter(ifs_y, ifs_x[:, f])
            plt.savefig('farf_scatter_%s_v_reward.png' % f)
            plt.close()
        exit()

    nn_stack.add(nn, support)

    for j in range(1, rec_steps + 1):
        # RESIDUALS MODEL #
        tic('Building SFADF dataset for residuals model')
        # State, features (stack), action, dynamics (nn), features (stack)
        sfadf = build_sfadf(nn_stack, nn, support, sars)
        F = pds_to_npa(sfadf.F)  # All features from NN stack
        D = pds_to_npa(sfadf.D)  # Feature dynamics of last NN
        log('Mean dynamic values %s' % np.mean(D, axis=0))
        toc()

        tic('Fitting residuals model')
        max_depth = F.shape[1]
        model = ExtraTreesRegressor(n_estimators=50,
                                    max_depth=max_depth)  # This should underfit
        model.fit(F, D)

        log('Cleaning memory (F, D)')
        del F, D
        toc()
        # END RESIDUALS MODEL #

        # NEURAL NETWORK > 0 #
        tic('Building SARes dataset')
        # Frames, action, residual dynamics of last NN (Res = D - model(F))
        sares = build_sares(model, sfadf)
        S = pds_to_npa(sares.S)  # 4 frames
        A = pds_to_npa(sares.A)  # Discrete action
        RES = pds_to_npa(sares.RES).squeeze()  # Residual dynamics of last NN
        log('Mean residual values %s' % np.mean(RES, axis=0))
        toc()

        tic('Fitting NN%s' % j)
        image_shape = S.shape[1:]
        target_size = RES.shape[1] if len(RES.shape) > 1 else 1
        # NN maps frames to residual dynamics of last NN
        nn = ConvNet(image_shape, target_size, nb_actions=nb_actions,
                     l1_alpha=0.0, nb_epochs=nn_nb_epochs)
        nn.fit(S, A, RES)
        nn.load('NN.h5')  # Load best network (saved by callback)

        log('Cleaning memory (sares, S, A, RES')
        del sares, S, A, RES
        toc()
        # END NEURAL NETWORK > 0 #

        # ITERATIVE FEATURE SELECTION > 0 #
        tic('Building FADF dataset for IFS')
        # Features (stack + last nn), action, dynamics (previous nn), features (stack + last nn)
        fadf = build_fadf(nn_stack, nn, sars, sfadf)
        ifs_x, ifs_y = split_dataset_for_ifs(fadf, features='F', target='D')
        toc()

        tic('Running IFS with target D')
        ifs = IFS(**ifs_params)
        preload_features = range(nn_stack.get_support_dim())
        ifs.fit(ifs_x, ifs_y, preload_features=preload_features)
        support = ifs.get_support()
        got_action = support[-1]
        support = support[len(preload_features):-1]  # Remove already selected features and action from support
        nb_new_features = np.array(support).sum()
        r2_change = (ifs.scores_[-1] - ifs.scores_[0]) / abs(ifs.scores_[0])
        log('IFS - New features: %s' % nb_new_features)
        log('Action was%s selected' % ('' if got_action else ' NOT'))
        log('R2 change %s (from %s to %s)' % (
        r2_change, ifs.scores_[0], ifs.scores_[-1]))

        log('Cleaning memory (fadf, ifs_x, ifs_y)')
        del fadf, ifs_x, ifs_y
        toc()
        # END ITERATIVE FEATURE SELECTION > 0 #

        nn_stack.add(nn, support)

        if nb_new_features == 0 or r2_change < r2_change_threshold:
            print 'Done.\n' + '#' * 50 + '\n'
            break

    # FITTED Q-ITERATION #
    tic('Building global FARF dataset for FQI')
    # Features (stack), action, reward, features (stack)
    global_farf = build_global_farf(nn_stack, sars)
    sast, r = split_dataset_for_fqi(global_farf)
    all_features_dim = nn_stack.get_support_dim()  # Need to pass new dimension of "states" to instantiate new FQI
    action_values = pds_to_npa(global_farf.A.unique())
    toc()

    tic('Updating policy')
    # Update ActionRegressor to only use the actions actually in the dataset
    regressor = Regressor(regressor_class=ExtraTreesRegressor,
                          **fqi_regressor_params)
    regressor = ActionRegressor(regressor,
                                discrete_actions=action_values,
                                tol=0.5,
                                **fqi_regressor_params)
    policy.fqi_params['estimator'] = regressor
    # TODO if it crashes, update policy.nn_stack here
    policy.fit_on_dataset(sast, r, all_features_dim)
    policy.epsilon_step()

    log('Cleaning memory (global_farf, sast, r)')
    del global_farf, sast, r
    toc()

    tic('Evaluating policy after update')
    evaluation_metrics = evaluate_policy(mdp, policy, max_ep_len=max_eval_steps)
    evaluation_results.append(evaluation_metrics)
    toc(evaluation_results)
    # END FITTED Q-ITERATION #

# FINAL OUTPUT #
# Plot evaluation results
evaluation_results = pd.DataFrame(evaluation_results,
                                  columns=['Score', 'Confidence (score)',
                                           'Steps', 'Confidence (steps)'])
evaluation_results[['Score', 'Steps']].plot().get_figure().savefig(
    logger.path + 'evaluation.png')

# TODO Log run configuration
# END #
