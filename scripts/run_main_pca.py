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
from deep_ifs.extraction.ConvNetClassifier import ConvNetClassifier
from deep_ifs.models.epsilonFQI import EpsilonFQI
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
parser.add_argument('--fqi-model-type', type=str, default='extra',
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
parser.add_argument('--sars-episodes', type=int, default=300,
                    help='Number of SARS episodes to collect')
parser.add_argument('--collect-gfarf', action='store_true',
                    help='Collect GFARF instead of generating it from SARS')
parser.add_argument('--gfarf-episodes', type=int, default=1000,
                    help='Number of global FARF episodes to collect')
parser.add_argument('--control-freq', type=int, default=2,
                    help='Control frequency (1 action every n steps)')
parser.add_argument('--initial-rg', type=float, default=1.,
                    help='Initial random/greedy split for collecting SARS\'')
parser.add_argument('--nn0l1', type=float, default=0.01,
                    help='l1 normalization for NN0')
parser.add_argument('--balanced-weights', action='store_true',
                    help='Use balanced weights instead of the custom ones')
parser.add_argument('--fqi-iter', type=int, default=300,
                    help='Number of FQI iterations to run')
parser.add_argument('--fqi-eval-period', type=int, default=5,
                    help='Number of FQI iterations between evaluations')
args = parser.parse_args()
# fqi-model and nn-stack must be both None or both set
assert not ((args.fqi_model is not None) ^ (args.nn_stack is not None)), \
    'Set both or neither --fqi-model and --nn-stack.'
# END ARGS

# HYPERPARAMETERS
sars_episodes = 10 if args.debug else args.sars_episodes  # Number of SARS episodes to collect
nn_nb_epochs = 2 if args.debug else 300  # Number of training epochs for NNs
algorithm_steps = 100  # Number of steps to make in the main loop
rec_steps = 1 if args.debug else 2  # Number of recursive steps to make
variance_pctg = 0.5  # Remove this many % of non-zero feature during FS (kinda)
fqi_iterations = 2 if args.debug else 120  # Number of steps to train FQI
eval_episodes = 1 if args.debug else 4  # Number of evaluation episodes to run
max_eval_steps = 2 if args.debug else 500  # Maximum length of eval episodes
random_greedy_step = 0.2  # Decrease R/G split by this much at each step
final_random_greedy_split = 0.1
random_greedy_split = args.initial_rg
fqi_patience = 15  # Number of FQI iterations w/o improvement after which to stop
fqi_iter = 5 if args.debug else args.fqi_iter  # Number of FQI iterations
fqi_eval_period = args.fqi_eval_period  # Number of FQI iterations after which to evaluate
initial_actions = [1, 4, 5]  # Initial actions for BreakoutDeterministic-v3

# SETUP
logger = Logger(output_folder='../output/',
                custom_run_name='run_pca%Y%m%d-%H%M%S')
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

# Create policy
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
    log('Loading NN stack from %s' % args.nn_stack)
    nn_stack.load(args.nn_stack)
    fqi_params = {'estimator': regressor,
                  'state_dim': nn_stack.get_support_dim(),
                  'action_dim': 1,  # Action is discrete monodimensional
                  'discrete_actions': action_values,
                  'gamma': mdp.gamma,
                  'horizon': fqi_iterations,
                  'verbose': True}
    log('Loading policy from %s' % args.fqi_model)
    policy = EpsilonFQI(fqi_params, nn_stack, fqi=args.fqi_model)
    evaluation_metrics = evaluate_policy(mdp,
                                         policy,
                                         max_ep_len=max_eval_steps,
                                         n_episodes=3,
                                         initial_actions=initial_actions)
    log('Loaded policy evaluation: %s' % str(evaluation_metrics))
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
                        initial_actions=initial_actions,
                        repeat=args.control_freq)

    # TODO NN analysis
    if args.nn_analysis:
        test_split = int(0.1 * len(sars))
        test_sars = sars[test_split:]
        sars = sars[:test_split]

    sars.to_pickle(logger.path + 'sars_%s.pickle' % i)  # Save SARS

    if args.collect_gfarf:
        tic('Collecting SARS to train FQI')
        sars_path = logger.path + 'sars_%s/' % i
        os.mkdir(sars_path)
        collect_sars_to_disk(mdp,
                             policy,
                             sars_path,
                             episodes=args.gfarf_episodes,
                             random_greedy_split=random_greedy_split,
                             debug=args.debug,
                             initial_actions=initial_actions,
                             repeat=args.control_freq)
        toc()

    S = pds_to_npa(sars.S)  # 4 frames
    A = pds_to_npa(sars.A)  # Discrete action
    R = pds_to_npa(sars.R)  # Scalar reward
    if args.clip_nn0:
        R = np.clip(R, -1, 1)  # Clipped scalar reward

    # TODO NN analysis
    if args.nn_analysis:
        test_S = pds_to_npa(test_sars.S)
        test_A = pds_to_npa(test_sars.A)
        test_R = pds_to_npa(test_sars.R)

    if args.balanced_weights:
        sars_sample_weight = get_sample_weight(R)
    else:
        class_weight = {-100: 100,
                        -1: 100,
                        0: 1,
                        1: 100,
                        4: 100,
                        7: 100}
        sars_sample_weight = get_sample_weight(R, class_weight=class_weight,
                                               round=True)

    log('Got %s SARS\' samples' % len(sars))
    log('Memory usage: %s MB' % get_size([sars, S, A, R], 'MB'))
    toc()

    tic('Resetting NN stack')
    nn_stack.reset()  # Clear the stack after collecting SARS' with last policy
    toc('Policy stack outputs %s features' % policy.nn_stack.get_support_dim())

    tic('Fitting NN0')
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
                               binarize=args.binarize,
                               logger=logger,
                               chkpt_file='NN0_step%s.h5' % i)
    else:
        target_size = 1  # Initial target is the scalar reward
        nn = ConvNet(mdp.state_shape,
                     target_size,
                     nb_actions=nb_actions,
                     l1_alpha=args.nn0l1,
                     sample_weight=sars_sample_weight,
                     nb_epochs=nn_nb_epochs,
                     binarize=args.binarize,
                     logger=logger,
                     chkpt_file='NN0_step%s.h5' % i)

    nn.fit(S, A, R)

    # TODO NN analysis
    if args.nn_analysis:
        pred = nn.predict(test_S, test_A)
        plt.suptitle('NN0 step %s' % i)
        plt.xlabel('Reward')
        plt.ylabel('NN prediction')
        plt.scatter(test_R, pred, alpha=0.3)
        plt.savefig(logger.path + 'NN0_step%s_R.png' % i)
        plt.close()
        del test_A, test_S, test_R

    del S, A, R
    nn.load(logger.path + 'NN0_step%s.h5' % i)  # Load best network (saved by callback)
    toc()

    # FEATURE SELECTION 0 #
    tic('Building F dataset for PCA')
    F = build_features(nn, sars)  # Features

    if args.clip_nn0:
        ifs_y = np.clip(ifs_y, -1, 1)

    nonzero_mfv_counts = np.count_nonzero(np.mean(F, axis=0))
    log('Number of non-zero feature: %s' % nonzero_mfv_counts)
    log('Memory usage: %s MB' % get_size([F], 'MB'))
    toc()

    tic('Applying PCA')
    v = np.var(F, axis=0, dtype=np.float64)
    del F
    v_uniq = np.unique(v)  # Sort and keep unique
    start = int(round(len(v_uniq) * variance_pctg))
    if start == len(v_uniq):
        log('Got bad features (by default all are kept, but there is probably'
            'something wrong with NN0).\nUnique variances array: %s' % v_uniq)
        support = np.repeat([True], len(v))
    else:
        variance_thresh = v_uniq[start:].min()
        log('Unique variances array: %s' % v_uniq)
        log('Variance threshold: %s' % variance_thresh)
        support = v > variance_thresh
        nb_new_features = support.sum()
        log('Features:\n%s' % support.nonzero())
        log('PCA - New features: %s' % nb_new_features)
    toc()

    # TODO Debug
    if args.debug:
        support[2] = True

    nn_stack.add(nn, support)

    for j in range(1, rec_steps + 1):
        # RESIDUALS MODEL #
        tic('Building SFAD dataset for residuals model')
        # State, features (stack), action, dynamics (nn), features (stack)
        sfad = build_sfad(nn_stack, nn, support, sars)
        F = pds_to_npa(sfad.F)  # All features from NN stack
        D = pds_to_npa(sfad.D)  # Feature dynamics of last NN
        log('Mean dynamic values %s' % np.mean(D, axis=0))
        log('Dynamic values variance %s' % np.std(D, axis=0))
        log('Max dynamic values %s' % np.max(D, axis=0))
        log('Memory usage: %s MB' % get_size([sfad, F, D], 'MB'))
        toc()

        tic('Fitting residuals model')
        if args.residual_model == 'extra':
            max_depth = 1 + F.shape[1] / 2
            model = ExtraTreesRegressor(n_estimators=50,
                                        max_depth=max_depth,
                                        n_jobs=-1)
        elif args.residual_model == 'linear':
            model = LinearRegression(n_jobs=-1)

        # Train residuals model
        model.fit(F, D, sample_weight=sars_sample_weight)
        del F, D
        toc()

        # NEURAL NETWORK i #
        tic('Building SARes dataset')
        # Frames, action, residual dynamics of last NN (Res = D - model(F))
        sares = build_sares(model, sfad)
        S = pds_to_npa(sares.S)  # 4 frames
        A = pds_to_npa(sares.A)  # Discrete action
        if args.no_residuals:
            log('Ignoring residuals, using only dynamics.')
            RES = pds_to_npa(sfad.D)  # Residuals are actually the dynamics
        else:
            RES = pds_to_npa(sares.RES)  # Residuals of last NN

        # TODO NN analysis
        if args.nn_analysis:
            test_S = pds_to_npa(test_sars.S)
            test_A = pds_to_npa(test_sars.A)
            test_RES = pds_to_npa(test_sars.RES)

        del sfad, sares
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
                     binarize=args.binarize,
                     scaler=StandardScaler(),
                     logger=logger,
                     chkpt_file='NN%s_step%s.h5' % (j, i))
        nn.fit(S, A, RES)

        # TODO NN analysis
        if args.nn_analysis:
            pred = nn.predict(test_S, test_A)
            for f in range(target_size):
                plt.figure()
                plt.suptitle('NN%s step %s' % (j, i))
                plt.xlabel('Residual feature %s of %s' % (f, target_size))
                plt.ylabel('NN prediction')
                if target_size > 1:
                    plt.scatter(test_RES[:, f], pred[:, f], alpha=0.3)
                else:
                    # Will only run the loop once
                    plt.scatter(test_RES[:], pred[:], alpha=0.3)
                plt.savefig(logger.path + 'NN%s_step%s_res%s.png' % (j, i, f))
                plt.close()
            del test_A, test_S, test_R

        del S, A, RES
        nn.load(logger.path + 'NN%s_step%s.h5' % (j, i))  # Load best network (saved by callback)
        toc()

        # FEATURE SELECTION i #
        tic('Building F dataset for PCA %s' % j)
        # Features (last nn)
        F = build_features(nn, sars)
        # Print the number of nonzero features
        nonzero_mfv_counts = np.count_nonzero(np.mean(F, axis=0))
        log('Number of non-zero feature: %s' % nonzero_mfv_counts)
        log('Memory usage: %s MB' % get_size([F], 'MB'))
        toc()

        tic('Applying PCA %s' % j)
        v = np.var(F, axis=0, dtype=np.float64)
        del F
        v_uniq = np.unique(v)  # Sort and keep unique
        start = int(round(len(v_uniq) * variance_pctg))
        if start == len(v_uniq):
            log('Got bad features. Unique variances array: %s' % v_uniq)
            toc()
            log('Done.\n')
            break
        else:
            variance_thresh = v_uniq[start:].min()
            log('Unique variances array: %s' % v_uniq)
            log('Variance threshold: %s' % variance_thresh)
            support = v > variance_thresh
            nb_new_features = support.sum()
            log('Features:\n%s' % support.nonzero())
            log('PCA - New features: %s' % nb_new_features)
        toc()

        nn_stack.add(nn, support)

        if nb_new_features == 0:
            log('Done.\n')
            break

    # FITTED Q-ITERATION #
    tic('Building global FARF dataset for FQI')
    # Features (stack), action, reward, features (stack)
    global_farf = build_global_farf(nn_stack, sars)
    del sars

    if args.collect_gfarf:
        global_farf_2 = build_global_farf_from_disk(nn_stack, sars_path)
        global_farf = global_farf.append(global_farf_2, ignore_index=True)
        log('Got %s additional FARF samples for a total of %s'
            % (len(global_farf_2), len(global_farf)))

    sast, r = split_dataset_for_fqi(global_farf)
    all_features_dim = nn_stack.get_support_dim()  # Need to pass new dimension of "states" to instantiate new ActionRegressor
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
    regressor = ActionRegressor(Regressor(regressor_class=fqi_regressor_class,
                                          **fqi_regressor_params),
                                discrete_actions=action_values,
                                tol=0.5)
    policy.fqi_params['estimator'] = regressor

    # Update policy using early stopping
    fqi_current_patience = fqi_patience
    fqi_best = (-np.inf, 0, -np.inf, 0)

    policy.reset(all_features_dim)
    policy.partial_fit(sast, r)
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
                                            append_filename='fqi_step_%03d_iter_%03d' % (i, partial_iter))
            policy.save_fqi(logger.path + 'fqi_step_%03d_iter_%03d_score_%s.pkl'
                            % (i, partial_iter, round(fqi_best[0])))
            log('Evaluation: %s' % str(es_evaluation))
            if es_evaluation[0] > fqi_best[0]:
                log('Saving best policy')
                fqi_best = es_evaluation
                fqi_current_patience = fqi_patience
                # Save best policy to restore it later
                policy.save_fqi(logger.path + 'best_fqi_%03d_score_%s.pkl'
                                % (i, round(fqi_best[0])))
            else:
                fqi_current_patience -= 1
                if fqi_current_patience == 0:
                    break

    # Restore best policy
    policy.load_fqi(logger.path + 'best_fqi_%03d_score_%s.pkl'
                    % (i, round(fqi_best[0])))

    # Decrease R/G split
    if random_greedy_split - random_greedy_step >= final_random_greedy_split:
        random_greedy_split -= random_greedy_step
    else:
        random_greedy_split = final_random_greedy_split

    del sast, r
    gc.collect()
    toc()

    tic('Evaluating best policy after update')
    evaluation_metrics = evaluate_policy(mdp,
                                         policy,
                                         max_ep_len=max_eval_steps,
                                         n_episodes=eval_episodes,
                                         save_video=args.save_video,
                                         save_path=logger.path,
                                         append_filename='best_step_%03d' % i,
                                         initial_actions=initial_actions)
    evaluation_results.append(evaluation_metrics)
    toc(evaluation_results)

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

toc('Done. Exit...')
# END #
