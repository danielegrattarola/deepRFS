"""
Algorithm pseudo-code

Definitions:
    NN[i]: is the i-th neural network and provides the features(S) method that
        returns all 512 features produced when state S is given as input to the
        network and the the s_features(S) method that returns all **selected**
        features
        produced when state S is given as input to each network.

    NN_stack: contains all trained neural networks so far, provides the
    s_features(S) method that returns all **selected** features produced when
    state S is given as input to each network.


Policy = fully random

Main loop:
    Collect SARS' samples with policy
    Fix dataset to account for imbalance (proportional to the number of
    transitions for each reward class)

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
import argparse
import gc
import os
from deep_ifs.envs.atari import Atari
from deep_ifs.evaluation.evaluation import *
from deep_ifs.extraction.NNStack import NNStack
from deep_ifs.extraction.ConvNet import ConvNet
from deep_ifs.models.epsilonFQI import EpsilonFQI
from deep_ifs.selection.ifs import IFS
from deep_ifs.utils.datasets import *
from deep_ifs.utils.Logger import Logger
from deep_ifs.utils.timer import *
from ifqi.models import Regressor, ActionRegressor
from matplotlib import pyplot as plt
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import LinearRegression, Ridge

# ARGS
parser = argparse.ArgumentParser()
parser.add_argument('-d', '--debug', action='store_true', help='Run in debug mode')
parser.add_argument('--save-video', action='store_true', help='Save the gifs of the evaluation episodes')
parser.add_argument('-e', '--env', type=str, default='BreakoutDeterministic-v3', help='Atari environment on which to run the algorithm')
parser.add_argument('--farf-analysis', action='store_true', help='Plot and save info about each FARF dataset generated during the run')
parser.add_argument('--residual-model', type=str, default='linear', help='Type of model to use for building residuals (\'linear\', \'extra\')')
parser.add_argument('--fqi-model-type', type=str, default='extra', help='Type of model to use for fqi (\'linear\', \'ridge\', \'extra\')')
parser.add_argument('--fqi-model', type=str, default=None, help='Path to a saved FQI pickle file to load as policy in the first iteration')
parser.add_argument('--nn-stack', type=str, default=None, help='Path to a saved NNStack folder to load as feature extractor in the first iteration')
parser.add_argument('--binarize', action='store_true', help='Binarize input to the neural networks')
args = parser.parse_args()
# fqi-model and nn-stack must be both None or both set
assert not ((args.fqi_model is not None) ^ (args.nn_stack is not None)), 'Set both or neither --fqi-model and --nn-stack.'
# END ARGS

# HYPERPARAMETERS
sars_episodes = 10 if args.debug else 200  # Number of SARS episodes to collect
nn_nb_epochs = 2 if args.debug else 300  # Number of training epochs for NNs
algorithm_steps = 100  # Number of steps to make in the main loop
rec_steps = 1 if args.debug else 100  # Number of recursive steps to make
ifs_nb_trees = 50  # Number of trees to use in IFS
ifs_significance = 1  # Significance for IFS
fqi_iterations = 2 if args.debug else 120  # Number of steps to train FQI
r2_change_threshold = 0.10  # % of IFS R2 improvement below which to stop loop
eval_episodes = 1 if args.debug else 4  # Number of evaluation episodes to run
max_eval_steps = 2 if args.debug else 500  # Maximum length of eval episodes
initial_random_greedy_split = 1  # Initial R/G split for SARS collection
random_greedy_step = 0.2  # Decrease R/G split by this much at eacch step
final_random_greedy_split = 0.1
random_greedy_split = initial_random_greedy_split
es_patience = 15  # Number of FQI iterations w/o improvement after which to stop
es_iter = 150  # Number of FQI iterations
es_eval_freq = 5  # Number of FQI iterations
initial_actions = [1, 4, 5]  # Initial actions for BreakoutDeterministic-v3

# SETUP
logger = Logger(output_folder='../output/', custom_run_name='run_ifs%Y%m%d-%H%M%S')
evaluation_results = []
nn_stack = NNStack()  # To store all neural networks and IFS supports
mdp = Atari(args.env)
action_values = mdp.action_space.values
nb_actions = mdp.action_space.n

# Create policy
# Create ActionRegressor
if args.fqi_model_type == 'extra':
    fqi_regressor_params = {'n_estimators': 50,
                            'n_jobs': -1}
    regressor = ActionRegressor(Regressor(regressor_class=ExtraTreesRegressor,
                                          **fqi_regressor_params),
                                discrete_actions=action_values,
                                tol=0.5)
elif args.fqi_model_type == 'linear':
    fqi_regressor_params = {'n_jobs': -1}
    regressor = ActionRegressor(Regressor(regressor_class=LinearRegression,
                                          **fqi_regressor_params),
                                discrete_actions=action_values,
                                tol=0.5)
elif args.fqi_model_type == 'ridge':
    fqi_regressor_params = {'n_jobs': -1}
    regressor = ActionRegressor(Regressor(regressor_class=Ridge,
                                          **fqi_regressor_params),
                                discrete_actions=action_values,
                                tol=0.5)
else:
    raise NotImplementedError('Allowed models: \'extra\', \'linear\', '
                              '\'ridge\'.')

# Create EpsilonFQI
if args.fqi_model is not None and args.nn_stack is not None:
    log('Loading NN stack from %s' % args.nn_stack)
    nn_stack.load(args.nn_stack)
    log('Loading policy from %s' % args.fqi_model)
    policy = EpsilonFQI(None, nn_stack, fqi=args.fqi_model)
    random_greedy_split = final_random_greedy_split
else:
    fqi_params = {'estimator': regressor,
                  'state_dim': nn_stack.get_support_dim(),
                  'action_dim': 1,  # Action is discrete monodimensional
                  'discrete_actions': action_values,
                  'gamma': mdp.gamma,
                  'horizon': fqi_iterations,
                  'verbose': True}
    policy = EpsilonFQI(fqi_params, nn_stack)  # Do not unpack the dict


log('######## START ########')
for i in range(algorithm_steps):
    # NEURAL NETWORK 0 #
    log('######## STEP %s ########' % i)

    tic('Collecting SARS dataset')
    # 4 frames, action, reward, 4 frames
    sars = collect_sars(mdp,
                        policy,
                        episodes=sars_episodes,
                        debug=args.debug,
                        random_greedy_split=random_greedy_split,
                        initial_actions=initial_actions)
    sars.to_pickle(logger.path + 'sars_%s.pickle' % i)  # Save SARS
    sars_sample_weight = get_sample_weight(sars)
    S = pds_to_npa(sars.S)  # 4 frames
    A = pds_to_npa(sars.A)  # Discrete action
    R = pds_to_npa(sars.R)  # Scalar reward
    log('Got %s SARS\' samples' % len(sars))
    log('Memory usage: %s MB' % get_size([sars, S, A, R], 'MB'))
    toc()

    tic('Resetting NN stack')
    nn_stack.reset()  # Clear the stack after collecting sars' with last policy
    toc('Policy stack outputs %s features' % policy.nn_stack.get_support_dim())

    tic('Fitting NN0')
    target_size = 1  # Initial target is the scalar reward
    # NN maps frames to reward
    nn = ConvNet(mdp.state_shape,
                 target_size,
                 nb_actions=nb_actions,
                 l1_alpha=0.01,
                 sample_weight=sars_sample_weight,
                 nb_epochs=nn_nb_epochs,
                 binarize=args.binarize)
    nn.fit(S, A, R)
    del S, A, R
    nn.load('NN.h5')  # Load best network (saved by callback)
    toc()

    # ITERATIVE FEATURE SELECTION 0 #
    tic('Building FARF dataset for IFS')
    farf = build_farf(nn, sars)  # Features, action, reward, next_features
    ifs_x, ifs_y = split_dataset_for_ifs(farf, features='F', target='R')
    del farf  # Not used anymore
    ifs_y = ifs_y.reshape(-1, 1)  # Sklearn version < 0.19 will throw a warning
    # Print the number of nonzero features
    nonzero_mfv_counts = np.count_nonzero(np.mean(ifs_x[:-1], axis=0))
    log('Number of non-zero feature: %s' % nonzero_mfv_counts)
    log('Memory usage: %s MB' % get_size([ifs_x, ifs_y], 'MB'))
    toc()

    tic('Running IFS with target R')
    ifs_estimator_params = {'n_estimators': ifs_nb_trees,
                            'n_jobs': -1}
    ifs_params = {'estimator': ExtraTreesRegressor(**ifs_estimator_params),
                  'n_features_step': 1,
                  'cv': None,
                  'scale': True,
                  'verbose': 0,
                  'significance': ifs_significance}
    ifs = IFS(**ifs_params)
    ifs.fit(ifs_x, ifs_y)
    support = ifs.get_support()
    got_action = support[-1]  # Action is the last feature
    support = support[:-1]  # Remove action from support
    nb_new_features = np.array(support).sum()
    r2_change = (ifs.scores_[-1] - ifs.scores_[0]) / abs(ifs.scores_[0])
    log('Features:\n%s' % np.array(support).nonzero())
    log('IFS - New features: %s' % nb_new_features)
    log('Action was%s selected' % ('' if got_action else ' NOT'))
    log('R2 change %s (from %s to %s)' % (r2_change, ifs.scores_[0], ifs.scores_[-1]))
    toc()

    # TODO Debug
    if args.debug:
        support[2] = True
    # TODO farf analysis
    if args.farf_analysis:
        feature_idxs = np.argwhere(support).reshape(-1)
        log('Mean feature values\n%s' % np.mean(ifs_x[:, feature_idxs], axis=0))
        for f in feature_idxs:
            np.save(logger.path + 'farf_feature_%s.npy' % f,
                    ifs_x[:, f].reshape(-1))
            plt.figure()
            plt.scatter(ifs_y.reshape(-1), ifs_x[:, f].reshape(-1))
            plt.savefig(logger.path + 'farf_scatter_%s_v_reward.png' % f)
            plt.close()

    del ifs_x, ifs_y
    nn_stack.add(nn, support)

    for j in range(1, rec_steps + 1):
        # RESIDUALS MODEL #
        tic('Building SFADF dataset for residuals model')
        # State, features (stack), action, dynamics (nn), features (stack)
        sfadf = build_sfadf(nn_stack, nn, support, sars)
        F = pds_to_npa(sfadf.F)  # All features from NN stack
        D = pds_to_npa(sfadf.D)  # Feature dynamics of last NN
        log('Mean dynamic values %s' % np.mean(D, axis=0))
        log('Dynamic values variance %s' % np.std(D, axis=0))
        log('Max dynamic values %s' % np.max(D, axis=0))
        log('Memory usage: %s MB' % get_size([sfadf, F, D], 'MB'))
        toc()

        tic('Fitting residuals model')
        if args.residual_model == 'extra':
            max_depth = F.shape[1] / 2
            model = ExtraTreesRegressor(n_estimators=50,
                                        max_depth=max_depth,
                                        n_jobs=-1)
        elif args.residual_model == 'linear':
            model = LinearRegression(n_jobs=-1)
        model.fit(F, D, sample_weight=sars_sample_weight)
        del F, D
        toc()

        # NEURAL NETWORK i #
        tic('Building SARes dataset')
        # Frames, action, residual dynamics of last NN (Res = D - model(F))
        sares = build_sares(model, sfadf)
        S = pds_to_npa(sares.S)  # 4 frames
        A = pds_to_npa(sares.A)  # Discrete action
        RES = pds_to_npa(sares.RES).squeeze()  # Residual dynamics of last NN
        del sares
        log('Mean residual values %s' % np.mean(RES, axis=0))
        log('Residual values variance %s' % np.std(RES, axis=0))
        log('Max residual values %s' % np.max(RES, axis=0))
        log('Memory usage: %s MB' % get_size([S, A, RES], 'MB'))
        toc()

        tic('Fitting NN%s' % j)
        image_shape = S.shape[1:]
        target_size = RES.shape[1] if len(RES.shape) > 1 else 1
        # NN maps frames to residual dynamics of last NN
        nn = ConvNet(image_shape,
                     target_size,
                     nb_actions=nb_actions,
                     l1_alpha=0.0,
                     sample_weight=sars_sample_weight,
                     nb_epochs=nn_nb_epochs,
                     binarize=args.binarize)
        nn.fit(S, A, RES)
        del S, A, RES
        nn.load('NN.h5')  # Load best network (saved by callback)
        toc()

        # ITERATIVE FEATURE SELECTION i #
        tic('Building FADF dataset for IFS %s' % j)
        # Features (stack + last nn), action, dynamics (previous nn), features (stack + last nn)
        fadf = build_fadf(nn_stack, nn, sars, sfadf)
        del sfadf
        ifs_x, ifs_y = split_dataset_for_ifs(fadf, features='F', target='D')
        del fadf
        log('Memory usage: %s MB' % get_size([ifs_x, ifs_y], 'MB'))
        toc()

        tic('Running IFS %s with target D' % j)
        ifs = IFS(**ifs_params)
        preload_features = range(nn_stack.get_support_dim())
        ifs.fit(ifs_x, ifs_y, preload_features=preload_features)
        del ifs_x, ifs_y
        support = ifs.get_support()
        got_action = support[-1]
        support = support[len(preload_features):-1]  # Remove already selected features and action from support
        nb_new_features = np.array(support).sum()
        r2_change = (ifs.scores_[-1] - ifs.scores_[0]) / abs(ifs.scores_[0])
        log('IFS - New features: %s' % nb_new_features)
        log('Action was%s selected' % ('' if got_action else ' NOT'))
        log('R2 change %s (from %s to %s)' % (r2_change, ifs.scores_[0], ifs.scores_[-1]))
        toc()
        # END ITERATIVE FEATURE SELECTION i #

        nn_stack.add(nn, support)

        if nb_new_features == 0 or r2_change < r2_change_threshold:
            log('Done.\n')
            break

    # FITTED Q-ITERATION #
    tic('Building global FARF dataset for FQI')
    # Features (stack), action, reward, features (stack)
    global_farf = build_global_farf(nn_stack, sars)
    del sars
    sast, r = split_dataset_for_fqi(global_farf)
    all_features_dim = nn_stack.get_support_dim()  # Need to pass new dimension of "states" to instantiate new FQI
    action_values = np.unique(pds_to_npa(global_farf.A))
    log('Memory usage: %s MB' % get_size([sast, r], 'MB'))
    toc()

    # Save dataset
    tic('Saving global FARF and NNStack')
    global_farf.to_pickle(logger.path + 'global_farf_%s.pickle' % i)
    del global_farf
    # Save nn_stack
    os.mkdir(logger.path + 'nn_stack_%s/' % i)
    nn_stack.save(logger.path + 'nn_stack_%s/' % i)
    toc()

    tic('Updating policy %s' % i)
    nb_reward_features = nn_stack.get_support_dim(index=0)
    nb_dynamics_features = nn_stack.get_support_dim() - nb_reward_features
    log('%s reward features' % nb_reward_features)
    log('%s dynamics features' % nb_dynamics_features)

    # Update ActionRegressor to only use the actions actually in the dataset
    if args.fqi_model_type == 'extra':
        fqi_regressor_params = {'n_estimators': 50}
        regressor = ActionRegressor(Regressor(regressor_class=ExtraTreesRegressor,
                                              **fqi_regressor_params),
                                    discrete_actions=action_values,
                                    tol=0.5)
    elif args.fqi_model_type == 'linear':
        regressor = ActionRegressor(Regressor(regressor_class=LinearRegression),
                                    discrete_actions=action_values,
                                    tol=0.5)
    elif args.fqi_model_type == 'ridge':
        regressor = ActionRegressor(Regressor(regressor_class=Ridge),
                                    discrete_actions=action_values,
                                    tol=0.5)
    policy.fqi_params['estimator'] = regressor

    # Update policy using early stopping
    es_current_patience = es_patience
    es_best = (-np.inf, 0, -np.inf, 0)

    policy.set_state_dim(all_features_dim)
    policy.partial_fit(sast, r)
    for partial_iter in range(es_iter):
        policy.partial_fit()
        if partial_iter % es_eval_freq == 0 or partial_iter == (es_iter - 1):
            es_evaluation = evaluate_policy(mdp,
                                            policy,
                                            max_ep_len=max_eval_steps,
                                            n_episodes=3,
                                            save_path=logger.path,
                                            initial_actions=initial_actions)
            log('Evaluation: %s' % str(es_evaluation))
            if es_evaluation[0] > es_best[0]:
                log('Saving best policy')
                es_best = es_evaluation
                es_current_patience = es_patience
                # Save best policy to restore it later
                policy.save_fqi(logger.path + 'best_fqi_%s_score_%s.pkl' % (i, round(es_best[0])))
            else:
                es_current_patience -= 1
                if es_current_patience == 0:
                    break

    # Restore best policy
    policy.load_fqi(logger.path + 'best_fqi_%s_score_%s.pkl' % (i, round(es_best[0])))

    # Decrease R/G split
    if random_greedy_split - random_greedy_step >= final_random_greedy_split:
        random_greedy_split -= random_greedy_step
    else:
        random_greedy_split = final_random_greedy_split

    del sast, r
    gc.collect()
    toc()

    tic('Evaluating policy after update')
    evaluation_metrics = evaluate_policy(mdp,
                                         policy,
                                         max_ep_len=max_eval_steps,
                                         n_episodes=eval_episodes,
                                         save_video=args.save_video,
                                         save_path=logger.path,
                                         append_filename='step_%s' % i,
                                         initial_actions=initial_actions)
    evaluation_results.append(evaluation_metrics)
    toc(evaluation_results)
    # END FITTED Q-ITERATION #

    log('######## DONE %s ########' % i + '\n')

# FINAL OUTPUT #
# Plot evaluation results
tic('Plotting evaluation results')
evaluation_results = pd.DataFrame(evaluation_results,
                                  columns=['score', 'confidence_score',
                                           'steps', 'confidence_steps'])
evaluation_results.to_csv('evaluation.csv', index=False)
fig = evaluation_results[['score', 'steps']].plot().get_figure()
fig.savefig(logger.path + 'evaluation.png')

# TODO Log run configuration
toc('Done. Exit...')
# END #
