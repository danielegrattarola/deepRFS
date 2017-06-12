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

import matplotlib

# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
from matplotlib import pyplot as plt

import joblib
import matplotlib
import gc

# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
import argparse
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

# ARGS
parser = argparse.ArgumentParser()
parser.add_argument('-d', '--debug', action='store_true',
                    help='Run in debug mode')
parser.add_argument('--save-video', action='store_true',
                    help='Save the gifs of the evaluation episodes')
parser.add_argument('-e', '--env', type=str, default='BreakoutDeterministic-v3',
                    help='Atari environment on which to run the algorithm')
parser.add_argument('--load-ae', type=str, default=None,
                    help='Path to h5 weights file to load into AE')
parser.add_argument('--fqi-model-type', type=str, default='xgb',
                    help='Type of model to use for fqi (\'linear\', \'ridge\', '
                         '\'extra\', \'xgb\')')
parser.add_argument('--fqi-model', type=str, default=None,
                    help='Path to a saved FQI pickle file to load as policy in '
                         'the first iteration')
parser.add_argument('--farf-analysis', action='store_true',
                    help='Plot and save info about each FARF dataset generated '
                         'during the run')
parser.add_argument('--binarize', action='store_true',
                    help='Binarize input to the neural networks')
parser.add_argument('--clip', action='store_true', help='Clip reward of MDP')
parser.add_argument('--use-actions', action='store_true',
                    help='Use actions to train the networks')
parser.add_argument('--load-sars', type=str, default=None,
                    help='Path to dataset folder to use instead of collecting')
parser.add_argument('--sars-episodes', type=int, default=100,
                    help='Number of SARS episodes to collect')
parser.add_argument('--sars-test-episodes', type=int, default=250,
                    help='Number of SARS test episodes to collect')
parser.add_argument('--sars-to-disk', type=int, default=25,
                    help='Number of SARS episodes to collect to disk')
parser.add_argument('--control-freq', type=int, default=1,
                    help='Control refrequency (1 action every n steps)')
parser.add_argument('--fqi-iter', type=int, default=300,
                    help='Number of FQI iterations to run')
parser.add_argument('--fqi-eval-period', type=int, default=1,
                    help='Number of FQI iterations between evaluations')
parser.add_argument('--no-fs', action='store_true',
                    help='RFS has no effect and all features are selected')
args = parser.parse_args()
# END ARGS

# HYPERPARAMETERS
sars_episodes = 10 if args.debug else args.sars_episodes  # Number of SARS episodes to collect
sars_test_episodes = 10 if args.debug else args.sars_test_episodes  # Number of SARS test episodes to collect
nn_nb_epochs = 5 if args.debug else 300  # Number of training epochs for NNs
nn_batch_size = 6 if args.debug else 32  # Number of samples in a batch for AE (len(sars) will be multiple of this number)
nn_encoding_dim = 512
ifs_nb_trees = 50  # Number of trees to use in IFS
ifs_significance = 1  # Significance for IFS
eval_episodes = 1 if args.debug else 4  # Number of evaluation episodes to run
max_eval_steps = 2 if args.debug else 500  # Maximum length of eval episodes
fqi_iter = 5 if args.debug else args.fqi_iter  # Number of FQI iterations
fqi_patience = fqi_iter  # Number of FQI iterations w/o improvement after which to stop
fqi_eval_period = args.fqi_eval_period  # Number of FQI iterations after which to evaluate
initial_actions = [1, 4, 5]  # Initial actions for BreakoutDeterministic-v3

# SETUP
logger = Logger(output_folder='../output/',
                custom_run_name='ae_rfs%Y%m%d-%H%M%S')
setup_logging(logger.path + 'log.txt')
log('LOCALS')
loc = locals().copy()
log('\n'.join(['%s, %s' % (k, v) for k, v in loc.iteritems()
               if not str(v).startswith('<')]))
log('\n')

evaluation_results = []
mdp = Atari(args.env, clip_reward=args.clip)
action_values = mdp.action_space.values
nb_actions = mdp.action_space.n if args.use_actions else 1

# Create ActionRegressor
if args.fqi_model_type == 'extra':
    fqi_regressor_params = {'n_estimators': 50,
                            'n_jobs': -1}
    fqi_regressor_class = ExtraTreesRegressor
elif args.fqi_model_type == 'xgb':
    fqi_regressor_params = {'max_depth': 10}
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

# NN: S -> R
target_size = 1  # Initial target is the scalar reward
ae = Autoencoder((4, 108, 84),
                 nb_epochs=nn_nb_epochs,
                 encoding_dim=nn_encoding_dim,
                 binarize=args.binarize,
                 logger=logger,
                 ckpt_file='autoencoder_ckpt.h5')

log(ae.model.summary())

# Create EpsilonFQI
fqi_params = {'estimator': regressor,
              'state_dim': nn_encoding_dim,
              'action_dim': 1,  # Action is discrete monodimensional
              'discrete_actions': action_values,
              'gamma': mdp.gamma,
              'horizon': fqi_iter,
              'verbose': True}
policy = EpsilonFQI(fqi_params, ae)  # Do not unpack the dict


log('######## START ########')

tic('Collecting SARS dataset')
if args.load_sars is None:
    # 4 frames, action, reward, 4 frames
    sars_path = logger.path + 'sars/'
    samples_in_dataset = collect_sars_to_disk(mdp,
                                              policy,
                                              sars_path,
                                              datasets=args.sars_to_disk,
                                              episodes=sars_episodes,
                                              debug=args.debug,
                                              random_greedy_split=1.0,
                                              initial_actions=initial_actions,
                                              repeat=args.control_freq,
                                              batch_size=nn_batch_size)
else:
    sars_path = args.load_sars
    samples_in_dataset = get_nb_samples_from_disk(sars_path)

toc('Got %s SARS\' samples' % samples_in_dataset)

# Collect test dataset
tic('Collecting test SARS dataset')
if args.load_sars is None:
    test_sars = collect_sars(mdp,
                             policy,
                             episodes=sars_test_episodes,
                             debug=args.debug,
                             random_greedy_split=1.0,
                             initial_actions=initial_actions,
                             repeat=args.control_freq)
else:
    test_sars = np.load(sars_path + '/valid_sars.npy')

test_S = pds_to_npa(test_sars[:, 0])
test_A = pds_to_npa(test_sars[:, 1])
test_R = pds_to_npa(test_sars[:, 2])
test_SS = pds_to_npa(test_sars[:, 3])
toc('Got %s test SARS\' samples' % len(test_sars))

log('Memory usage (test_sars, test_S, test_A, test_R, test_SS): %s MB\n' %
    get_size([test_sars, test_S, test_A, test_R, test_SS], 'MB'))

# Fit AE
if args.load_ae is None:
    tic('Fitting Autoencoder')
    ss_generator = ss_generator_from_disk(sars_path,
                                          ae,
                                          batch_size=nn_batch_size,
                                          binarize=args.binarize)
    ae.fit_generator(ss_generator,
                     samples_in_dataset / nn_batch_size,
                     nn_nb_epochs,
                     validation_data=(test_S, test_SS))
    ae.load(logger.path + 'autoencoder_ckpt.h5')
else:
    tic('Loading AE from %s' % args.load_ae)
    ae.load(args.load_ae)

del test_sars, test_S, test_A, test_R
gc.collect()

toc()

# RFS
tic('Building dataset for RFS')
F, A, R, FF = build_farf_from_disk(ae, sars_path)

# Print the number of nonzero features
toc('Number of non-zero feature: %s' % np.count_nonzero(np.mean(F[:-1], axis=0)))

if args.no_fs:
    support = np.var(F[:, :-1], axis=0) != 0  # Keep only features with nonzero variance
else:
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

    gc.collect()

    # Process support
    support = rfs.get_support()
    got_action = support[-1]  # Action is the last feature
    support = support[:-1]  # Remove action from support
    nb_new_features = np.array(support).sum()
    log('Features: %s' % np.array(support).nonzero())
    log('IFS - New features: %s' % nb_new_features)
    log('Action was%s selected' % ('' if got_action else ' NOT'))

    # Save RFS tree
    tree = rfs.export_graphviz(filename=logger.path + 'rfs_tree.gv')
    tree.save()  # Save GV source
    tree.format = 'pdf'
    tree.render()  # Save PDF

    if args.farf_analysis:
        feature_idxs = np.argwhere(support).reshape(-1)
        log('Mean feature values\n%s\n' % np.mean(F[:, feature_idxs], axis=0))
        for f in feature_idxs:
            np.save(logger.path + 'farf_feature_%s.npy' % f,
                    F[:, f].reshape(-1))
            plt.figure()
            plt.scatter(R.reshape(-1), F[:, f].reshape(-1), alpha=0.3)
            plt.savefig(logger.path + 'farf_scatter_%s_v_reward.png' % f)
            plt.close()

    del F, A, FF, R

ae.set_support(support)

# FITTED Q-ITERATION
tic('Building dataset for FQI')
faft, r, action_values = build_faft_r_from_disk(ae, sars_path)
toc('Got %s samples' % len(faft))

# Save dataset
log('Saving global FARF')
joblib.dump((faft, r, action_values), logger.path + 'global_farf.pickle')

# Update policy parameters
new_state_dim = support.sum()
new_action_values = np.unique(faft[:, new_state_dim])
regressor = ActionRegressor(Regressor(regressor_class=fqi_regressor_class,
                                      **fqi_regressor_params),
                            discrete_actions=new_action_values,
                            tol=0.5)
fqi_params = {'estimator': regressor,
              'state_dim': new_state_dim,
              'action_dim': 1,  # Action is discrete monodimensional
              'discrete_actions': new_action_values,
              'gamma': mdp.gamma,
              'horizon': fqi_iter,
              'verbose': True}
policy = EpsilonFQI(fqi_params, ae)  # Do not unpack the dict

# Update policy using early stopping
fqi_current_patience = fqi_patience
fqi_best = (-np.inf, 0, -np.inf, 0)

policy.partial_fit(faft, r)
for partial_iter in range(fqi_iter):
    policy.partial_fit()
    if partial_iter % fqi_eval_period == 0 or partial_iter == (fqi_iter - 1):
        es_evaluation = evaluate_policy(mdp,
                                        policy,
                                        n_episodes=3,
                                        initial_actions=initial_actions,
                                        save_video=args.save_video,
                                        save_path=logger.path,
                                        append_filename='fqi_iter_%03d' % partial_iter)
        evaluation_results.append(es_evaluation )
        policy.save_fqi(logger.path + 'fqi_iter_%03d_score_%s.pkl'
                        % (partial_iter, round(fqi_best[0])))
        log('Evaluation: %s' % str(es_evaluation))
        if es_evaluation[0] > fqi_best[0]:
            log('Saving best policy')
            fqi_best = es_evaluation
            fqi_current_patience = fqi_patience
            # Save best policy to restore it later
            policy.save_fqi(logger.path + 'best_fqi_score_%s.pkl'
                            % round(fqi_best[0]))
        else:
            fqi_current_patience -= 1
            if fqi_current_patience == 0:
                break

# Restore best policy
policy.load_fqi(logger.path + 'best_fqi_score_%s.pkl'
                % round(fqi_best[0]))

toc()

tic('Evaluating best policy after update')
evaluation_metrics = evaluate_policy(mdp,
                                     policy,
                                     n_episodes=eval_episodes,
                                     save_video=args.save_video,
                                     save_path=logger.path,
                                     append_filename='best',
                                     initial_actions=initial_actions)
toc(evaluation_metrics)

# FINAL OUTPUT #
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
