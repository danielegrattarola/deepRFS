import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

import joblib
import gc
import argparse
import atexit
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import LinearRegression, Ridge
from xgboost import XGBRegressor
from deep_ifs.envs.atari import Atari
from deep_ifs.evaluation.evaluation import *
from deep_ifs.extraction.Autoencoder import Autoencoder
from deep_ifs.models.epsilonFQI import EpsilonFQI
from deep_ifs.selection.ifs import IFS
from deep_ifs.selection.rfs import RFS
from deep_ifs.utils.datasets import *
from deep_ifs.utils.Logger import Logger
from deep_ifs.utils.timer import *
from deep_ifs.utils.helpers import get_size
from ifqi.models import Regressor, ActionRegressor
from sklearn.neural_network import MLPRegressor


def plot_output():
    # Writes evaluation results to csv and saves plots
    global evaluation_results, main_alg_iter
    output = pd.DataFrame(evaluation_results,
                          columns=['score', 'score_max', 'confidence_score',
                                   'steps', 'steps_max', 'confidence_steps'])
    output.to_csv(logger.path + 'evaluation_%s.csv' % main_alg_iter, index=False)
    fig = output['score'].plot().get_figure()
    fig.savefig(logger.path + 'evaluation_score_%s.png' % main_alg_iter)
    plt.close()
    fig = output['score_max'].plot().get_figure()
    fig.savefig(logger.path + 'evaluation_score_max_%s.png' % main_alg_iter)
    plt.close()
    fig = output['steps'].plot().get_figure()
    fig.savefig(logger.path + 'evaluation_steps_%s.png' % main_alg_iter)
    plt.close()
    fig = output['steps_max'].plot().get_figure()
    fig.savefig(logger.path + 'evaluation_steps_max_%s.png' % main_alg_iter)
    plt.close()

atexit.register(plot_output)

# Args
# Main
parser = argparse.ArgumentParser()
parser.add_argument('-d', '--debug', action='store_true', help='Run in debug mode')
parser.add_argument('--main-alg-iters', type=int, default=1, help='Number of main algorithm steps to run (by default runs in full batch mode)')

# MDP
parser.add_argument('-e', '--env', type=str, default='BreakoutDeterministic-v4', help='Atari environment on which to run the algorithm')
parser.add_argument('--clip', action='store_true', help='Clip reward of MDP')
parser.add_argument('--clip-eval', action='store_true', help='Clip reward of MDP during evaluation')

# AE
parser.add_argument('--load-ae', type=str, default=None, help='Path to h5 weights file to load into AE')
parser.add_argument('--load-ae-support', type=str, default=None, help='Path to file with AE support')
parser.add_argument('--train-ae', action='store_true', help='Train the AE after collecting the dataset')
parser.add_argument('--ae-epochs', type=int, default=300, help='Number of epochs to train AE for')
parser.add_argument('--binarize', action='store_true', help='Binarize input to the neural networks')
parser.add_argument('--use-sw', action='store_true', help='Use sample weights when training AE')
parser.add_argument('--use-vae', action='store_true', help='Use VAE instead of usual AE')
parser.add_argument('--vae-beta', type=float, default=1., help='Beta hyperparameter for Beta-VAE')
parser.add_argument('--use-dense', action='store_true', help='Use AE with dense inner layer instead of usual AE')
parser.add_argument('--dropout', type=float, default=0., help='Dropout rate for dense AE')
parser.add_argument('--n-features', type=int, default=128, help='Number of features for contractive, dense and VAE')

# RFS
parser.add_argument('--fs', action='store_true', help='Select features')
parser.add_argument('--rfs', action='store_true', help='Use RFS to select features (otherwise all non-zero variance features are kept)')

# FQI
parser.add_argument('--load-fqi', type=str, default=None, help='Path to fqi file to load into policy')
parser.add_argument('--fqi-initial-epsilon', type=float, default=1, help='Initial exploration rate for FQI')
parser.add_argument('--fqi-load-faft', type=str, help='Load FAFT, R and action values for FQI from file')
parser.add_argument('--fqi-model-type', type=str, default='extra', help='Type of model to use for fqi (\'linear\', \'ridge\', \'extra\', \'xgb\')')
parser.add_argument('--fqi-no-ar', action='store_true', help='Do not use ActionRegressor')
parser.add_argument('--fqi-iter', type=int, default=5000, help='Number of FQI iterations to run')
parser.add_argument('--fqi-eval-episodes', type=int, default=2, help='Number of episodes to evaluate FQI')
parser.add_argument('--fqi-eval-period', type=int, default=1, help='Number of FQI iterations after which to evaluate')
parser.add_argument('--save-video', action='store_true', help='Save the gifs of the evaluation episodes')
parser.add_argument('--fqi-test-after-loading', action='store_true', help='Test FQI after loading it')

# Dataset cocllection
parser.add_argument('--load-sars', type=str, default=None, help='Path to dataset folder to use instead of collecting')
parser.add_argument('--sars-samples', type=int, default=500000, help='Number of SARS samples to collect')
parser.add_argument('--sars-blocks', type=int, default=25, help='Number of SARS episodes to collect to disk')
parser.add_argument('--sars-test-episodes', type=int, default=100, help='Number of SARS test episodes to collect')
parser.add_argument('--force-valid-sars', action='store_true', help='Force the collection of a validation SARS')
parser.add_argument('--control-freq', type=int, default=1, help='Control refrequency (1 action every n steps)')
parser.add_argument('--save-FARF', action='store_true', help='Save the F, A, R, FF arrays')
parser.add_argument('--load-FARF', type=str, default=None, help='Load the F, A, R, FF arrays')

args = parser.parse_args()

# Parameters
# Env
initial_actions = [1]  # Initial actions for BreakoutDeterministic-v4

# AE
nn_nb_epochs = 5 if args.debug else args.ae_epochs  # Number of training epochs for NNs
nn_batch_size = 6 if args.debug else 32  # Number of samples in a batch for AE (len(sars) will be multiple of this number)
nn_binarization_threshold = 0.35 if args.env == 'PongDeterministic-v4' else 0.1

# RFS
ifs_nb_trees = 50  # Number of trees to use in IFS
ifs_significance = 1  # Significance for IFS

# FQI
epsilon = args.fqi_initial_epsilon
epsilon_min = 0.1
epsilon_step = (epsilon - epsilon_min) / args.main_alg_iters

# Run setup
rn_list = []
if args.train_ae:
    rn_list.append('ae')
else:
    rn_list.append('fqi')
if args.fs:
    if args.rfs:
        rn_list.append('rfs')
    else:
        rn_list.append('nzv')
rn_list.append('%Y%m%d-%H%M%S')
custom_run_name = '_'.join(rn_list)
logger = Logger(output_folder='../output/', custom_run_name=custom_run_name)
setup_logging(logger.path + 'log.txt')

# Environment
mdp = Atari(args.env, clip_reward=args.clip)
action_values = mdp.action_space.values

# Autoencoder (this one will be used as FE, but never trained)
ae = Autoencoder((4, 108, 84),
                 n_features=args.n_features,
                 batch_size=nn_batch_size,
                 nb_epochs=nn_nb_epochs,
                 binarize=args.binarize,
                 binarization_threshold=nn_binarization_threshold,
                 logger=logger,
                 ckpt_file='autoencoder_ckpt_0.h5',
                 use_vae=args.use_vae,
                 beta=args.vae_beta,
                 use_dense=args.use_dense,
                 dropout_prob=args.dropout)
ae.model.summary()
if args.load_ae is None:
    support = np.array([True] * ae.get_features_number())  # Keep all features
else:
    ae.load(args.load_ae)
    if args.load_ae_support is not None:
        support = joblib.load(args.load_ae_support)  # Load support from previous training
    else:
        support = np.array([True] * ae.get_features_number())  # Keep all features
ae.set_support(support)

# Create EpsilonFQI
if args.load_fqi is None:
    # Don't care, will only be used as fully random policy and never trained
    fqi_regressor_params = {'n_jobs': -1}
    fqi_regressor_class = LinearRegression
    regressor = Regressor(regressor_class=fqi_regressor_class,
                          **fqi_regressor_params)
    fqi_params = {'estimator': regressor,
                  'state_dim': ae.get_features_number(),
                  'action_dim': 1,  # Action is discrete monodimensional
                  'discrete_actions': action_values,
                  'gamma': mdp.gamma,
                  'horizon': args.fqi_iter,
                  'verbose': True}
    policy = EpsilonFQI(fqi_params, ae, epsilon=epsilon)  # Here epsilon = initial epsilon
else:
    fqi_params = args.load_fqi
    policy = EpsilonFQI(fqi_params, ae, epsilon=epsilon)

    if args.fqi_test_after_loading:
        # Evaluate policy after loading
        partial_eval = evaluate_policy(mdp,
                                       policy,
                                       n_episodes=5,
                                       initial_actions=initial_actions,
                                       save_video=args.save_video,
                                       save_path=logger.path,
                                       append_filename='fqi_test_after_loading',
                                       eval_epsilon=0.05,
                                       clip=args.clip_eval)
        log('Testing FQI after loading: %s' % str(partial_eval))

# Log locals
log('LOCALS')
loc = locals().copy()
log('\n'.join(['%s, %s' % (k, v) for k, v in loc.iteritems()
               if not str(v).startswith('<')]) + '\n')

log('######## START ########')
for main_alg_iter in range(args.main_alg_iters):
    if args.load_sars is None or main_alg_iter > 0:
        tic('Collecting SARS dataset')
        sars_path = logger.path + 'sars_%s/' % main_alg_iter
        samples_in_dataset = collect_sars_to_disk(mdp,
                                                  policy,
                                                  sars_path,
                                                  samples=args.sars_samples,
                                                  blocks=args.sars_blocks,
                                                  debug=args.debug,
                                                  random_episodes_pctg=0.0,
                                                  initial_actions=initial_actions,
                                                  repeat=args.control_freq,
                                                  batch_size=nn_batch_size,
                                                  shuffle=False)
    else:
        tic('Loading SARS dataset from disk')
        sars_path = args.load_sars
        samples_in_dataset = get_nb_samples_from_disk(sars_path)
    toc('Got %s SARS\' samples' % samples_in_dataset)

    if args.train_ae or main_alg_iter > 0:
        # Collect test dataset
        if args.load_sars is None or args.force_valid_sars:
            tic('Collecting test SARS dataset')
            test_sars = collect_sars(mdp,
                                     policy,
                                     episodes=args.sars_test_episodes,
                                     debug=args.debug,
                                     random_episodes_pctg=0.0,
                                     initial_actions=initial_actions,
                                     repeat=args.control_freq,
                                     shuffle=False)
            np.save(sars_path + 'valid_sars.npy', test_sars)
        else:
            tic('Loading test SARS from disk')
            test_sars = np.load(sars_path + 'valid_sars.npy')

        test_S = pds_to_npa(test_sars[:, 0])
        toc('Got %s test SARS\' samples' % len(test_sars))

        log('Memory usage (test_sars, test_S): %s MB\n' %
            get_size([test_sars, test_S, ae], 'MB'))

        if args.load_ae is not None or main_alg_iter > 0:
            # Reset AE after collecting samples with old AE
            ae = Autoencoder((4, 108, 84),
                             n_features=args.n_features,
                             batch_size=nn_batch_size,
                             nb_epochs=nn_nb_epochs,
                             binarize=args.binarize,
                             binarization_threshold=nn_binarization_threshold,
                             logger=logger,
                             ckpt_file='autoencoder_ckpt_%s.h5' % main_alg_iter,
                             use_vae=args.use_vae,
                             beta=args.vae_beta,
                             use_dense=args.use_dense,
                             dropout_prob=args.dropout)

        # Fit AE
        tic('Fitting Autoencoder')
        if args.use_sw:
            tic('Getting class weights')
            cw = get_class_weight_from_disk(sars_path, clip=args.clip)
            toc(cw)
        else:
            cw = None
        ss_generator = ss_generator_from_disk(sars_path,
                                              ae,
                                              batch_size=nn_batch_size,
                                              binarize=args.binarize,
                                              binarization_threshold=nn_binarization_threshold,
                                              weights=cw,
                                              shuffle=True,
                                              clip=args.clip)
        ae.fit_generator(ss_generator,
                         samples_in_dataset / nn_batch_size,
                         nn_nb_epochs,
                         validation_data=(test_S, test_S))
        ae.load(logger.path + 'autoencoder_ckpt_%s.h5' % main_alg_iter)

        del test_sars, test_S
        gc.collect()
        toc()

    if args.fs:
        # Feature selection
        if args.load_FARF is None:
            tic('Building FARF dataset for FS')
            F, A, R, FF = build_farf_from_disk(ae, sars_path, shuffle=True)
            if args.save_FARF:
                joblib.dump((F, A, R, FF), logger.path + 'RFS_F_A_R_F_%s.pkl' % main_alg_iter)
        else:
            tic('Loading FARF dataset for FS from %s' % args.load_FARF)
            F, A, R, FF = joblib.load(args.load_FARF)

        if args.clip:
            R = np.clip(R, -1, 1)

        toc('Number of non-zero feature: %s' % np.count_nonzero(np.mean(F[:-1], axis=0)))

        if args.rfs:
            tic('Running RFS')
            ifs_estimator_params = {'n_estimators': ifs_nb_trees,
                                    'n_jobs': -1}
            ifs_params = {'estimator': ExtraTreesRegressor(**ifs_estimator_params),
                          'n_features_step': 1,
                          'cv': None,
                          'scale': True,
                          'verbose': 1,
                          'significance': ifs_significance}
            ifs = IFS(**ifs_params)
            features_names = np.array(map(str, range(F.shape[1])) + ['A'])
            rfs_params = {'feature_selector': ifs,
                          'features_names': features_names,
                          'verbose': 1}
            rfs = RFS(**rfs_params)
            rfs.fit(F, A, FF, R)

            # Process support
            support = rfs.get_support()
            got_action = support[-1]  # Action is the last feature
            support = np.array(support[:-1])  # Remove action from support
            nb_new_features = support.sum()
            log('Features: %s' % support.nonzero())
            log('Using %s features' % nb_new_features)
            log('Action was%s selected' % ('' if got_action else ' NOT'))

            # Save RFS tree
            tree = rfs.export_graphviz(filename=logger.path + 'rfs_tree_%s.gv' % main_alg_iter)
            tree.save()  # Save GV source
            tree.format = 'pdf'
            tree.render()  # Save PDF

            del F, A, FF, R
            gc.collect()
            toc()
        else:
            tic('Keeping non-zero variance features')
            support = np.var(F, axis=0) != 0  # Keep only features with nonzero variance
            toc('Using %s features' % support.sum())

        ae.set_support(support)
        joblib.dump(support, logger.path + 'support_%s.pkl' % main_alg_iter)  # Save support

    # Build dataset for FQI
    if args.fqi_load_faft is None:
        tic('Building dataset for FQI')
        faft, r, action_values = build_faft_r_from_disk(ae, sars_path, shuffle=True)
        # Save dataset
        log('Saving dataset')
        joblib.dump((faft, r, action_values), logger.path + 'FQI_FAFT_R_action_values_%s.pkl' % main_alg_iter)
    else:
        tic('Loading dataset for FQI')
        faft, r, action_values = joblib.load(args.fqi_load_faft)
        log('Shuffling data')
        perm = np.random.permutation(len(faft))
        faft = faft[perm]
        r = r[perm]
        del perm

    if args.clip:
        r = np.clip(r, -1, 1)

    toc('Got %s samples' % len(faft))

    log('Creating policy')
    # Create ActionRegressor
    if args.fqi_model_type == 'extra':
        fqi_regressor_params = {'n_estimators': 50,
                                'min_samples_split': 5,
                                'min_samples_leaf': 2,
                                'n_jobs': -1}
        fqi_regressor_class = ExtraTreesRegressor
    elif args.fqi_model_type == 'xgb':
        fqi_regressor_params = {'max_depth': 8,
                                'n_estimators': 100}
        fqi_regressor_class = XGBRegressor
    elif args.fqi_model_type == 'linear':
        fqi_regressor_params = {'n_jobs': -1}
        fqi_regressor_class = LinearRegression
    elif args.fqi_model_type == 'mlp':
        fqi_regressor_params = {'hidden_layer_sizes': (128, 128),
                                'early_stopping': True}
        fqi_regressor_class = MLPRegressor
    else:
        raise NotImplementedError('Allowed models: \'extra\', \'linear\', \'xgb\', \'mlp\'.')

    if args.fqi_no_ar:
        regressor = Regressor(regressor_class=fqi_regressor_class,
                              **fqi_regressor_params)
    else:
        regressor = ActionRegressor(Regressor(regressor_class=fqi_regressor_class,
                                              **fqi_regressor_params),
                                    discrete_actions=action_values,
                                    tol=0.5)

    epsilon -= epsilon_step  # Linear epsilon annealing

    fqi_params = {'estimator': regressor,
                  'state_dim': ae.get_support_dim(),
                  'action_dim': 1,  # Action is discrete monodimensional
                  'discrete_actions': action_values,
                  'gamma': mdp.gamma,
                  'horizon': args.fqi_iter,
                  'verbose': False}
    policy = EpsilonFQI(fqi_params, ae, epsilon=epsilon)  # Do not unpack the dict

    # Fit FQI
    log('Fitting FQI')
    evaluation_results = []
    fqi_best = (-np.inf, 0, -np.inf, 0)

    policy.partial_fit(faft, r)
    for partial_iter in range(args.fqi_iter):
        policy.partial_fit()
        if partial_iter % args.fqi_eval_period == 0 or partial_iter == (args.fqi_iter-1):
            print 'Eval...'
            partial_eval = evaluate_policy(mdp,
                                           policy,
                                           n_episodes=args.fqi_eval_episodes,
                                           initial_actions=initial_actions,
                                           save_video=args.save_video,
                                           save_path=logger.path,
                                           append_filename='fqi_iter_%s_%03d' % (main_alg_iter, partial_iter),
                                           eval_epsilon=0.05,
                                           clip=args.clip_eval)
            evaluation_results.append(partial_eval)
            log('Iter %s: %s' % (partial_iter, evaluation_results[-1]))
            plot_output()
            # Save fqi policy
            if partial_eval[0] > fqi_best[0]:
                log('Saving best policy\n')
                fqi_best = partial_eval
                policy.save_fqi(logger.path + 'best_fqi_score_%s_%s.pkl' % (main_alg_iter, round(fqi_best[0])))

    # Restore best policy
    policy.load_fqi(logger.path + 'best_fqi_score_%s_%s.pkl' % (main_alg_iter, round(fqi_best[0])))

    # Final evaluation
    tic('Evaluating best policy after update')
    final_eval = evaluate_policy(mdp,
                                 policy,
                                 n_episodes=args.fqi_eval_episodes,
                                 save_video=args.save_video,
                                 save_path=logger.path,
                                 append_filename='best_%s' % main_alg_iter,
                                 initial_actions=initial_actions,
                                 eval_epsilon=0.05,
                                 clip=args.clip_eval)
    toc(final_eval)
