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
from deep_ifs.envs.atari import Atari
from deep_ifs.evaluation.evaluation import *
from deep_ifs.extraction.NNStack import NNStack
from deep_ifs.extraction.ConvNet import ConvNet
from deep_ifs.extraction.ConvNetClassifier import ConvNetClassifier
from deep_ifs.models.epsilonFQI import EpsilonFQI
from deep_ifs.selection.ifs import IFS
from deep_ifs.utils.datasets import *
from deep_ifs.utils.Logger import Logger
from deep_ifs.utils.timer import *
from deep_ifs.utils.helpers import get_size
from ifqi.models import Regressor, ActionRegressor
from matplotlib import pyplot as plt
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor


# ARGS
parser = argparse.ArgumentParser()
parser.add_argument('-d', '--debug', action='store_true',
                    help='Run in debug mode')
parser.add_argument('--save-video', action='store_true',
                    help='Save the gifs of the evaluation episodes')
parser.add_argument('-e', '--env', type=str, default='BreakoutDeterministic-v3',
                    help='Atari environment on which to run the algorithm')
parser.add_argument('--farf-analysis', action='store_true',
                    help='Plot and save info about each FARF dataset generated '
                         'during the run')
parser.add_argument('--nn-analysis', action='store_true',
                    help='Plot predictions for each network')
parser.add_argument('--residual-model', type=str, default='linear',
                    help='Type of model to use for building residuals (\'linear'
                         '\', \'extra\')')
parser.add_argument('--fqi-model-type', type=str, default='xgb',
                    help='Type of model to use for fqi (\'linear\', \'ridge\', '
                         '\'extra\', \'xgb\')')
parser.add_argument('--fqi-model', type=str, default=None,
                    help='Path to a saved FQI pickle file to load as policy in '
                         'the first iteration')
parser.add_argument('--nn-stack', type=str, default=None,
                    help='Path to a saved NNStack folder to load as feature '
                         'extractor in the first iteration')
parser.add_argument('--binarize', action='store_true',
                    help='Binarize input to the neural networks')
parser.add_argument('--classify', action='store_true',
                    help='Use a classifier for NN0')
parser.add_argument('--clip', action='store_true', help='Clip reward of MDP')
parser.add_argument('--clip-nn0', action='store_true',
                    help='Clip reward for NN0 only')
parser.add_argument('--no-residuals', action='store_true',
                    help='Ignore residuals model and use directly the dynamics')
parser.add_argument('--sars-episodes', type=int, default=500,
                    help='Number of SARS episodes to collect')
parser.add_argument('--sars-test-episodes', type=int, default=100,
                    help='Number of SARS test episodes to collect')
parser.add_argument('--sars-to-disk', type=int, default=1,
                    help='Number of SARS episodes to collect to disk')
parser.add_argument('--control-freq', type=int, default=2,
                    help='Control frequency (1 action every n steps)')
parser.add_argument('--initial-rg', type=float, default=1.,
                    help='Initial random/greedy split for collecting SARS\'')
parser.add_argument('--nn0l1', type=float, default=0.001,
                    help='l1 normalization for NN0')
parser.add_argument('--balanced-weights', action='store_true',
                    help='Use balanced weights instead of the custom ones')
parser.add_argument('--fqi-iter', type=int, default=60,
                    help='Number of FQI iterations to run')
parser.add_argument('--fqi-eval-period', type=int, default=1,
                    help='Number of FQI iterations between evaluations')
args = parser.parse_args()
# fqi-model and nn-stack must be both None or both set
assert not ((args.fqi_model is not None) ^ (args.nn_stack is not None)), \
    'Set both or neither --fqi-model and --nn-stack.'
# END ARGS

# HYPERPARAMETERS
sars_episodes = 10 if args.debug else args.sars_episodes  # Number of SARS episodes to collect
sars_test_episodes = 10 if args.debug else args.sars_test_episodes  # Number of SARS test episodes to collect
nn_nb_epochs = 5 if args.debug else 300  # Number of training epochs for NNs
nn_batch_size = 6 if args.debug else 32  # Number of samples in a batch for NN (len(sars) will be multiple of this number)
algorithm_steps = 100  # Number of steps to make in the main loop
rec_steps = 1 if args.debug else 100  # Number of recursive steps to make
ifs_nb_trees = 50  # Number of trees to use in IFS
ifs_significance = 1  # Significance for IFS
r2_change_threshold = 0.10  # % of IFS R2 improvement below which to stop loop
eval_episodes = 1 if args.debug else 4  # Number of evaluation episodes to run
max_eval_steps = 2 if args.debug else 500  # Maximum length of eval episodes
random_greedy_step = 0.2  # Decrease R/G split by this much at each step
final_random_greedy_split = 0.1
random_greedy_split = args.initial_rg
fqi_iter = 5 if args.debug else args.fqi_iter  # Number of FQI iterations
fqi_patience = fqi_iter  # Number of FQI iterations w/o improvement after which to stop
fqi_eval_period = args.fqi_eval_period  # Number of FQI iterations after which to evaluate
initial_actions = [1, 4, 5]  # Initial actions for BreakoutDeterministic-v3
class_weight = {-100: 100,
                -1: 100,
                0: 1,
                1: 100,
                4: 100,
                7: 100}

# SETUP
logger = Logger(output_folder='../output/',
                custom_run_name='run_ifs%Y%m%d-%H%M%S')
setup_logging(logger.path + 'log.txt')
log('LOCALS')
loc = locals().copy()
log('\n'.join(['%s, %s' % (k, v) for k, v in loc.iteritems()
               if not str(v).startswith('<')]))
log('\n')

evaluation_results = []
nn_stack = NNStack()  # To store all neural networks and FS supports
mdp = Atari(args.env, clip_reward=args.classify or args.clip)
action_values = mdp.action_space.values
nb_actions = mdp.action_space.n

# Create ActionRegressor
if args.fqi_model_type == 'extra':
    fqi_regressor_params = {'n_estimators': 50,
                            'n_jobs': -1}
    fqi_regressor_class = ExtraTreesRegressor
elif args.fqi_model_type == 'xgb':
    fqi_regressor_params = {}
    fqi_regressor_class = XGBRegressor
elif args.fqi_model_type == 'linear':
    fqi_regressor_params = {'n_jobs': -1}
    fqi_regressor_class = LinearRegression
elif args.fqi_model_type == 'ridge':
    fqi_regressor_params = {}
    fqi_regressor_class = Ridge
else:
    raise NotImplementedError('Allowed models: \'extra\', \'linear\', '
                              '\'ridge\', \'xgb\'.')

regressor = ActionRegressor(Regressor(regressor_class=fqi_regressor_class,
                                      **fqi_regressor_params),
                            discrete_actions=action_values,
                            tol=0.5)

# Create EpsilonFQI
if args.fqi_model is not None and args.nn_stack is not None:
    tic('Loading policy from %s' % args.fqi_model)
    log('Loading NN stack from %s' % args.nn_stack)
    nn_stack.load(args.nn_stack)
    fqi_params = {'estimator': regressor,
                  'state_dim': nn_stack.get_support_dim(),
                  'action_dim': 1,  # Action is discrete monodimensional
                  'discrete_actions': action_values,
                  'gamma': mdp.gamma,
                  'horizon': fqi_iter,
                  'verbose': True}
    policy = EpsilonFQI(fqi_params, nn_stack, fqi=args.fqi_model)
    evaluation_metrics = evaluate_policy(mdp,
                                         policy,
                                         max_ep_len=max_eval_steps,
                                         n_episodes=3,
                                         initial_actions=initial_actions)
    toc('Loaded policy - evaluation: %s' % str(evaluation_metrics))
else:
    fqi_params = {'estimator': regressor,
                  'state_dim': nn_stack.get_support_dim(),
                  'action_dim': 1,  # Action is discrete monodimensional
                  'discrete_actions': action_values,
                  'gamma': mdp.gamma,
                  'horizon': fqi_iter,
                  'verbose': True}
    policy = EpsilonFQI(fqi_params, nn_stack)  # Do not unpack the dict


log('######## START ########')
for step in range(algorithm_steps):
    log('######## STEP %s ########' % step)

    tic('Collecting SARS dataset')
    # 4 frames, action, reward, 4 frames
    sars_path = logger.path + 'sars_%s/' % step
    samples_in_dataset = collect_sars_to_disk(mdp,
                                              policy,
                                              sars_path,
                                              datasets=args.sars_to_disk,
                                              episodes=sars_episodes,
                                              debug=args.debug,
                                              random_greedy_split=random_greedy_split,
                                              initial_actions=initial_actions,
                                              repeat=args.control_freq,
                                              batch_size=nn_batch_size)
    toc('Got %s SARS\' samples' % samples_in_dataset)

    # Collect test dataset
    tic('Collecting test SARS dataset')
    test_sars = collect_sars(mdp,
                             policy,
                             episodes=sars_test_episodes,
                             debug=args.debug,
                             random_greedy_split=random_greedy_split,
                             initial_actions=initial_actions,
                             repeat=args.control_freq)
    test_S = pds_to_npa(test_sars[:, 0])
    test_A = pds_to_npa(test_sars[:, 1])
    test_R = pds_to_npa(test_sars[:, 2])

    test_sars_sample_weight = get_sample_weight(test_R,
                                                balanced=args.balanced_weights,
                                                class_weight=class_weight,
                                                round_target=True)

    toc('Got %s test SARS\' samples' % len(test_sars))

    log('Memory usage: %s MB\n' % get_size([test_sars, test_S, test_A, test_R], 'MB'))

    log('Resetting NN stack')
    nn_stack.reset()  # Clear the stack after collecting SARS' with last policy
    log('Policy stack outputs %s features\n' % policy.nn_stack.get_support_dim())

    # NN: S -> R
    if args.classify:
        from sklearn.preprocessing import OneHotEncoder
        ohe = OneHotEncoder(sparse=False)
        R = ohe.fit_transform(R.reshape(-1, 1) - R.min())
        nb_classes = R.shape[1]  # Target is the one-hot encoded reward
        nn = ConvNetClassifier(mdp.state_shape,
                               nb_classes,
                               nb_actions=nb_actions,
                               l1_alpha=0.0,
                               nb_epochs=nn_nb_epochs,
                               binarize=args.binarize,
                               logger=logger,
                               chkpt_file='NN0_step%s.h5' % step)
    else:
        target_size = 1  # Initial target is the scalar reward
        nn = ConvNet(mdp.state_shape,
                     target_size,
                     nb_actions=nb_actions,
                     l1_alpha=args.nn0l1,
                     nb_epochs=nn_nb_epochs,
                     binarize=args.binarize,
                     logger=logger,
                     chkpt_file='NN0_step%s.h5' % step)

    # Fit NN0
    tic('Fitting NN0 (target: R)')
    sar_generator = sar_generator_from_disk(sars_path,
                                            batch_size=nn_batch_size,
                                            balanced=args.balanced_weights,
                                            class_weight=class_weight,
                                            round_target=True,
                                            binarize=args.binarize,
                                            clip=args.clip_nn0)
    nn.fit_generator(sar_generator,
                     samples_in_dataset / nn_batch_size,
                     nn_nb_epochs,
                     validation_data=([test_S, test_A], test_R),
                     clip=args.clip_nn0)
    nn.load(logger.path + 'NN0_step%s.h5' % step)
    toc()

    # TODO NN analysis
    if args.nn_analysis:
        pred = nn.predict(test_S, test_A)
        plt.suptitle('NN0 step %s' % step)
        plt.xlabel('Reward')
        plt.ylabel('NN prediction')
        plt.scatter(test_R, pred, alpha=0.3)
        plt.savefig(logger.path + 'NN0_step%s_R.png' % step)
        plt.close()

    # ITERATIVE FEATURE SELECTION 0
    tic('Building dataset for IFS 0')
    FA, R = build_far_from_disk(nn,
                                sars_path,
                                clip=args.clip_nn0)  # Features, action, reward, next_features

    # Print the number of nonzero features
    toc('Number of non-zero feature: %s' %
        np.count_nonzero(np.mean(FA[:-1], axis=0)))

    log('Memory usage: %s MB\n' % get_size([FA, R], 'MB'))

    tic('Running IFS (target: R)')
    ifs_estimator_params = {'n_estimators': ifs_nb_trees,
                            'n_jobs': -1}
    ifs_params = {'estimator': ExtraTreesRegressor(**ifs_estimator_params),
                  'n_features_step': 1,
                  'cv': None,
                  'scale': True,
                  'verbose': 0,
                  'significance': ifs_significance}
    ifs = IFS(**ifs_params)

    # Fit IFS0
    ifs.fit(FA, R)

    # Process support
    support = ifs.get_support()
    got_action = support[-1]  # Action is the last feature
    support = support[:-1]  # Remove action from support
    nb_new_features = np.array(support).sum()
    r2_change = (ifs.scores_[-1] - ifs.scores_[0]) / abs(ifs.scores_[0])
    log('Features: %s' % np.array(support).nonzero())
    log('IFS - New features: %s' % nb_new_features)
    log('Action was%s selected' % ('' if got_action else ' NOT'))
    toc('R2 change %s (from %s to %s)' % (r2_change, ifs.scores_[0], ifs.scores_[-1]))

    # TODO Debug
    if args.debug:
        support = np.array([True, True] + [False] * 510)

    # TODO farf analysis
    if args.farf_analysis:
        feature_idxs = np.argwhere(support).reshape(-1)
        log('Mean feature values\n%s\n' % np.mean(FA[:, feature_idxs], axis=0))
        for f in feature_idxs:
            np.save(logger.path + 'farf_feature_%s.npy' % f,
                    FA[:, f].reshape(-1))
            plt.figure()
            plt.scatter(R.reshape(-1), FA[:, f].reshape(-1), alpha=0.3)
            plt.savefig(logger.path + 'farf_scatter_%s_v_reward.png' % f)
            plt.close()

    del FA, R

    # Add network to support
    nn_stack.add(nn, support)

    for i in range(1, rec_steps + 1):
        # Residual model
        if args.residual_model == 'extra':
            max_depth = 1 + nn_stack.get_support_dim() / 2
            model = ExtraTreesRegressor(n_estimators=50,
                                        max_depth=max_depth,
                                        n_jobs=-1)
        elif args.residual_model == 'linear':
            model = LinearRegression(n_jobs=-1)

        # Fit residuals model
        tic('Fitting residuals model')
        F, D = build_fd_from_disk(nn_stack, nn, support, sars_path)
        model.fit(F, D)
        toc()

        # Test data
        test_F, test_D = build_fd(nn_stack, nn, support, test_sars)
        test_RES = build_res(model, test_F, test_D, no_residuals=args.no_residuals)

        # Neural network
        image_shape = mdp.state_shape
        target_size = support.sum().astype('int32')
        nn = ConvNet(image_shape,
                     target_size,
                     nb_actions=nb_actions,
                     l1_alpha=0.0,
                     nb_epochs=nn_nb_epochs,
                     binarize=args.binarize,
                     logger=logger,
                     chkpt_file='NN%s_step%s.h5' % (i, step))

        log('Fitting scaler for residuals')
        scaler = fit_res_scaler(StandardScaler(),
                                F,
                                D,
                                model,
                                no_residuals=args.no_residuals)

        # Generator
        sares_generator = sares_generator_from_disk(model,
                                                    nn_stack,
                                                    nn_stack.get_model(-1),
                                                    nn_stack.get_support(-1),
                                                    sars_path,
                                                    batch_size=nn_batch_size,
                                                    scaler=scaler,
                                                    binarize=args.binarize,
                                                    no_residuals=args.no_residuals,
                                                    use_sample_weights=True,
                                                    balanced=args.balanced_weights,
                                                    class_weight=class_weight,
                                                    round_target=True,
                                                    clip=False)

        # Fit NNi (target: RES)
        tic('Fitting NN%s' % i)
        test_RES = scaler.transform(test_RES)  # Scale validation target
        nn.fit_generator(sares_generator,
                         samples_in_dataset / nn_batch_size,
                         nn_nb_epochs,
                         validation_data=([test_S, test_A], test_RES))
        nn.load(logger.path + 'NN%s_step%s.h5' % (i, step))
        toc()

        # TODO NN analysis
        if args.nn_analysis:
            pred = nn.predict(test_S, test_A)
            for f in range(target_size):
                plt.figure()
                plt.suptitle('NN%s step %s' % (i, step))
                plt.xlabel('Residual feature %s of %s' % (f, target_size))
                plt.ylabel('NN prediction')
                if target_size > 1:
                    plt.scatter(test_RES[:, f], pred[:, f], alpha=0.3)
                else:
                    # Will only run the loop once
                    plt.scatter(test_RES[:], pred[:], alpha=0.3)
                plt.savefig(logger.path + 'NN%s_step%s_res%s.png' % (i, step, f))
                plt.close()

        del test_A, test_S, test_RES

        # ITERATIVE FEATURE SELECTION i
        tic('Building FADF dataset for IFS %s' % i)
        # Features (stack + last nn), action, dynamics (previous nn), features (stack + last nn)
        FA = build_fa_from_disk(nn_stack, nn, sars_path)
        # D is already computed
        toc()

        log('Memory usage: %s MB\n' % get_size([FA, D], 'MB'))

        tic('Running IFS %s with target D' % i)
        ifs = IFS(**ifs_params)
        preload_features = range(nn_stack.get_support_dim())
        ifs.fit(FA, D, preload_features=preload_features)
        del FA, D

        support = ifs.get_support()
        got_action = support[-1]
        support = support[len(preload_features):-1]  # Remove already selected features and action from support
        nb_new_features = np.array(support).sum()
        r2_change = (ifs.scores_[-1] - ifs.scores_[0]) / abs(ifs.scores_[0])
        log('IFS - New features: %s' % nb_new_features)
        log('Action was%s selected' % ('' if got_action else ' NOT'))
        toc('R2 change %s (from %s to %s)' % (r2_change, ifs.scores_[0], ifs.scores_[-1]))

        nn_stack.add(nn, support)

        if nb_new_features == 0 or r2_change < r2_change_threshold:
            log('Feature extraction done.\n')
            break

    # FITTED Q-ITERATION
    tic('Building dataset for FQI')
    faft, r, action_values = build_fart_r_from_disk(nn_stack, sars_path)
    all_features_dim = nn_stack.get_support_dim()  # Pass new dimension of states to create ActionRegressor
    toc('Got %s samples' % len(faft))

    log('Memory usage: %s MB\n' % get_size([faft, r], 'MB'))

    # Save dataset
    tic('Saving global FARF and NNStack')
    joblib.dump((faft, r), logger.path + 'global_farf_%s.pickle' % step)
    # Save nn_stack
    os.mkdir(logger.path + 'nn_stack_%s/' % step)
    nn_stack.save(logger.path + 'nn_stack_%s/' % step)
    toc()

    tic('Updating policy %s' % step)
    nb_reward_features = nn_stack.get_support_dim(index=0)
    nb_dynamics_features = nn_stack.get_support_dim() - nb_reward_features
    log('%s reward features' % nb_reward_features)
    log('%s dynamics features' % nb_dynamics_features)

    # Update ActionRegressor to only use the actions actually in the dataset
    regressor = ActionRegressor(Regressor(regressor_class=fqi_regressor_class,
                                          **fqi_regressor_params),
                                discrete_actions=action_values,
                                tol=0.5)
    policy.fqi_params['estimator'] = regressor

    # Update policy using early stopping
    fqi_current_patience = fqi_patience
    fqi_best = (-np.inf, 0, -np.inf, 0)

    policy.reset(all_features_dim)
    policy.partial_fit(faft, r)
    for partial_iter in range(fqi_iter):
        policy.partial_fit()
        if partial_iter % fqi_eval_period == 0 or partial_iter == (fqi_iter - 1):
            es_evaluation = evaluate_policy(mdp,
                                            policy,
                                            max_ep_len=max_eval_steps,
                                            n_episodes=3,
                                            initial_actions=initial_actions,
                                            save_video=args.save_video,
                                            save_path=logger.path,
                                            append_filename='fqi_step_%03d_iter_%03d' % (step, partial_iter))
            policy.save_fqi(logger.path + 'fqi_step_%03d_iter_%03d_score_%s.pkl'
                            % (step, partial_iter, round(fqi_best[0])))
            log('Evaluation: %s' % str(es_evaluation))
            if es_evaluation[0] > fqi_best[0]:
                log('Saving best policy')
                fqi_best = es_evaluation
                fqi_current_patience = fqi_patience
                # Save best policy to restore it later
                policy.save_fqi(logger.path + 'best_fqi_%03d_score_%s.pkl'
                                % (step, round(fqi_best[0])))
            else:
                fqi_current_patience -= 1
                if fqi_current_patience == 0:
                    break

    # Restore best policy
    policy.load_fqi(logger.path + 'best_fqi_%03d_score_%s.pkl'
                    % (step, round(fqi_best[0])))

    # Decrease R/G split
    if random_greedy_split - random_greedy_step >= final_random_greedy_split:
        random_greedy_split -= random_greedy_step
    else:
        random_greedy_split = final_random_greedy_split
    toc()

    del faft, r
    gc.collect()

    tic('Evaluating best policy after update')
    evaluation_metrics = evaluate_policy(mdp,
                                         policy,
                                         max_ep_len=max_eval_steps,
                                         n_episodes=eval_episodes,
                                         save_video=args.save_video,
                                         save_path=logger.path,
                                         append_filename='best_step_%03d' % step,
                                         initial_actions=initial_actions)
    evaluation_results.append(evaluation_metrics)
    toc(evaluation_results)

    log('######## DONE %s ########' % step + '\n')

# FINAL OUTPUT #
# Plot evaluation results
tic('Plotting evaluation results')
evaluation_results = pd.DataFrame(evaluation_results,
                                  columns=['score', 'confidence_score',
                                           'steps', 'confidence_steps'])
evaluation_results.to_csv('evaluation.csv', index=False)
fig = evaluation_results['score'].plot().get_figure()
fig.savefig(logger.path + 'score_evaluation.png')
fig = evaluation_results['steps'].plot().get_figure()
fig.savefig(logger.path + 'steps_evaluation.png')
toc('Done. Exit...')
# END #
