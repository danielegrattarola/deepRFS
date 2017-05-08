import numpy as np
import pandas as pd
from deep_ifs.utils.helpers import flat2list, pds_to_npa, is_stuck
from joblib import Parallel, delayed
from tqdm import tqdm
from sklearn.utils.class_weight import compute_class_weight
import glob
import joblib
import os


# DATASET BUILDERS
def episode(mdp, policy, video=False, initial_actions=None, repeat=1):
    """
    Collects an episode of the given MDP using the given policy.

    Args
        mdp (Object): an mdp object (e.g. deep_ifs.envs.atari.Atari).
        policy (Object): a policy object (e.g. deep_ifs.models.EpsilonFQI).
            Methods draw_action and set_epsilon are expected.
        video (bool, False): render the video of the episode.
        initial_actions (list, None): list of action indices that start an
            episode of the MDP.

    Return
        A sequence of SARS' transitions as np.array
    """
    frame_counter = 0

    # Get current state
    state = mdp.reset()

    # Force start
    if initial_actions is not None:
        action = np.random.choice(initial_actions)
        state, _, _, _ = mdp.step(action)

    reward = 0
    done = False

    # Start episode
    ep_output = []
    while not done:
        frame_counter += 1

        # Select and execute the action, get next state and reward
        action = policy.draw_action(np.expand_dims(state, 0), done)
        action = int(action)
        # Repeat action
        temp_reward = 0
        temp_done = False
        for _ in range(repeat):
            next_state, reward, done, info = mdp.step(action)
            temp_reward += reward
            temp_done = temp_done or done
            if temp_done:
                break
        reward = temp_reward
        done = temp_done

        # build SARS' tuple
        ep_output.append([state, action, reward, next_state, done])

        # Render environment
        if video:
            mdp.render()

        # Update state
        state = next_state

    return ep_output


def collect_sars(mdp, policy, episodes=100, n_jobs=1, random_greedy_split=0.9,
                 debug=False, initial_actions=None, shuffle=True, repeat=1):
    """
    Collects a dataset of SARS' transitions of the given MDP.
    A percentage of the samples (random_greedy_split) is collected with a fully
    random policy, whereas the remaining part is collected with a greedy policy.

    Args
        mdp (Object): an mdp object (e.g. deep_ifs.envs.atari.Atari).
        policy (Object): a policy object (e.g. deep_ifs.models.EpsilonFQI).
            Methods draw_action and set_epsilon are expected.
        episodes (int, 100): number of episodes to collect.
        n_jobs (int, 1): number of processes to use (-1 for all available cores).
            Leave 1 if running stuff on GPU.
        random_greedy_split (float, 0.9): percentage of random episodes to
            collect.
        debug (bool, False): collect the episodes in debug mode (only a very
            small fraction of transitions will be returned).
        initial_actions (list, None): list of action indices that start an
            episode of the MDP.
        shuffle (bool, True): whether to shuffle the dataset before returning
            it.

    Return
        A SARS' dataset as pd.DataFrame with columns 'S', 'A', 'R', 'SS', 'DONE'
    """
    random_episodes = int(episodes * random_greedy_split)
    greedy_episodes = episodes - random_episodes

    policy.set_epsilon(1)
    dataset_random = Parallel(n_jobs=n_jobs)(
        delayed(episode)(mdp, policy, initial_actions=initial_actions,
                         repeat=repeat)
        for _ in tqdm(xrange(random_episodes))
    )
    # Each episode is in a list, so the dataset needs to be flattened
    dataset_random = np.asarray(flat2list(dataset_random))

    policy.set_epsilon(0)
    dataset_greedy = Parallel(n_jobs=n_jobs)(
        delayed(episode)(mdp, policy, initial_actions=initial_actions,
                         repeat=repeat)
        for _ in tqdm(xrange(greedy_episodes))
    )
    # Each episode is in a list, so the dataset needs to be flattened
    dataset_greedy = np.asarray(flat2list(dataset_greedy))

    if len(dataset_greedy) != 0 and len(dataset_random) != 0:
        dataset = np.append(dataset_random, dataset_greedy, 0)
    elif len(dataset_greedy) != 0:
        dataset = dataset_greedy
    else:
        dataset = dataset_random

    if shuffle:
        np.random.shuffle(dataset)

    # TODO debug
    if debug:
        dataset = dataset[:7]

    header = ['S', 'A', 'R', 'SS', 'DONE']
    return pd.DataFrame(dataset, columns=header)


def collect_sars_to_disk(mdp, policy, path, datasets=1, episodes=100,
                         n_jobs=1, random_greedy_split=0.9, debug=False,
                         initial_actions=None, shuffle=True, repeat=1):
    """
    Collects datasets of SARS' transitions of the given MDP and saves them to
    disk.
    A percentage of each dataset (random_greedy_split) is collected with a fully
    random policy, whereas the remaining part is collected with a greedy policy.

    Args
        mdp (Object): an mdp object (e.g. deep_ifs.envs.atari.Atari).
        policy (Object): a policy object (e.g. deep_ifs.models.EpsilonFQI).
            Methods draw_action and set_epsilon are expected.
        path (str): folder in which to save the dataset.
        datasets (int, 100): number of datasets to collect.
        episodes (int, 100): number of episodes in a dataset.
        n_jobs (int, 1): number of processes to use (-1 for all available cores).
            Leave 1 if running on GPU.
        random_greedy_split (float, 0.9): percentage of random episodes to
            collect.
        debug (bool, False): collect the episodes in debug mode (only a very
            small fraction of transitions will be returned).
        initial_actions (list, None): list of action indices that start an
            episode of the MDP.
        shuffle (bool, True): whether to shuffle the dataset before returning
            it.
        repeat (int, 1): control frequency for the policy.
    """
    if not path.endswith('/'):
        path += '/'
    if not os.path.exists(path):
        os.mkdir(path)

    for i in range(datasets):
        sars = collect_sars(mdp, policy, episodes=episodes, n_jobs=n_jobs,
                            random_greedy_split=random_greedy_split,
                            debug=debug, initial_actions=initial_actions,
                            shuffle=shuffle, repeat=repeat)
        sars.to_pickle(path + 'sars_%s.pkl' % i)


def sar_generator_from_disk(path):
    """
    Generator of S, A, R arrays from SARS datasets saved in path.
    
    Args
        path (str): path to folder containing 'sars_*.pkl' files (as collected
            with collect_sars_to_disk)
    
    Yield
        (S, A, R) (np.array, np.array, np.array): np.arrays with states, 
            actions and rewards from each SARS dataset in path.
     
    """
    if not path.endswith('/'):
        path += '/'
    files = glob.glob(path + 'sars_*.pkl')
    print 'Got %s files' % len(files)
    for f in files:
        sars = joblib.load(f)
        S = pds_to_npa(sars.S)
        A = pds_to_npa(sars.A)
        R = pds_to_npa(sars.R)
        del sars
        yield S, A, R


def sars_from_disk(path, datasets=1):
    """
    Args
        path (str): path to folder containing 'sars_*.pkl' files (as collected
            with collect_sars_to_disk)
        datasets (int, 1): number of datasets to read and merge.

    Return
        A single SARS' dataset composed of the first n datasets in path, 
        as pd.DataFrame with columns 'S', 'A', 'R', 'SS', 'DONE'
    """
    if not path.endswith('/'):
        path += '/'
    files = glob.glob(path + 'sars_*.pkl')
    files = files[:datasets]
    sars = pd.DataFrame()
    for f in files:
        sars = sars.append(joblib.load(f), ignore_index=True)
    return sars


def build_farf(nn, sars):
    """
    Builds FARF' dataset using SARS' dataset:
        F = NN[i].features(S)
        A = A
        R = R
        F' = NN[i].features(S')
    """
    header = ['F', 'A', 'R', 'FF']
    df = pd.DataFrame(columns=header)
    df['F'] = nn.all_features(pds_to_npa(sars.S)).tolist()
    df['A'] = sars.A
    df['R'] = sars.R
    df['FF'] = nn.all_features(pds_to_npa(sars.SS)).tolist()
    return df


def build_sfadf(nn_stack, nn, support, sars):
    """
    Builds SFADF' dataset using SARS' dataset:
        S = S
        F = NN_stack.s_features(S)
        A = A
        D = NN[i-1].s_features(S) - NN[i-1].s_features(S')
        F' = NN_stack.s_features(S')
    """
    header = ['S', 'F', 'A', 'D', 'FF']
    df = pd.DataFrame(columns=header)
    df['S'] = sars.S
    df['F'] = nn_stack.s_features(pds_to_npa(sars.S)).tolist()
    df['A'] = sars.A
    dynamics = nn.s_features(pds_to_npa(sars.S), support) - \
               nn.s_features(pds_to_npa(sars.SS), support)
    df['D'] = dynamics.tolist()
    df['FF'] = nn_stack.s_features(pds_to_npa(sars.SS)).tolist()
    return df


def build_sfad(nn_stack, nn, support, sars):
    """
    Builds SFAD dataset using SARS' dataset:
        S = S
        F = NN_stack.s_features(S)
        A = A
        D = NN[i-1].s_features(S) - NN[i-1].s_features(S')
    """
    header = ['S', 'F', 'A', 'D']
    df = pd.DataFrame(columns=header)
    df['S'] = sars.S
    df['F'] = nn_stack.s_features(pds_to_npa(sars.S)).tolist()
    df['A'] = sars.A
    dynamics = nn.s_features(pds_to_npa(sars.S), support) - \
               nn.s_features(pds_to_npa(sars.SS), support)
    df['D'] = dynamics.tolist()
    return df


def build_fadf(nn_stack, nn, sars, sfadf):
    """
    Builds new FADF' dataset from SARS' and SFADF':
        F = NN_stack.s_features(S) + NN[i].features(S)
        A = A
        D = SFADF'.D
        F' = NN_stack.s_features(S') + NN[i].features(S')
    """
    header = ['F', 'A', 'D', 'FF']
    df = pd.DataFrame(columns=header)
    features = np.column_stack((nn_stack.s_features(pds_to_npa(sars.S)),
                                nn.all_features(pds_to_npa(sars.S))))
    df['F'] = features.tolist()
    df['A'] = sars.A
    df['D'] = sfadf.D
    features = np.column_stack((nn_stack.s_features(pds_to_npa(sars.SS)),
                                nn.all_features(pds_to_npa(sars.SS))))
    df['FF'] = features.tolist()
    return df


def build_fadf_no_preload(nn, sars, sfadf):
    """
    Builds new FADF' dataset from SARS' and SFADF':
        F = NN[i].features(S)
        A = A
        D = SFADF'.D
        F' = NN[i].features(S')
    """
    header = ['F', 'A', 'D', 'FF']
    df = pd.DataFrame(columns=header)
    df['F'] = nn.all_features(pds_to_npa(sars.S)).tolist()
    df['A'] = sars.A
    df['D'] = sfadf.D
    df['FF'] = nn.all_features(pds_to_npa(sars.SS)).tolist()
    return df


def build_sares(model, sfadf, no_residuals=False):
    """
    Builds SARes dataset from SFADF':
        S = S
        A = A
        Res = D - M(F)
    """
    header = ['S', 'A', 'RES']
    df = pd.DataFrame(columns=header)
    df['S'] = sfadf.S
    df['A'] = sfadf.A
    dynamics = pds_to_npa(sfadf.D)
    features = pds_to_npa(sfadf.F)
    if no_residuals:
        df['RES'] = sfadf.D
    else:
        predictions = model.predict(features)
        residuals = dynamics - predictions
        df['RES'] = residuals.tolist()
    return df


def sares_generator_from_disk(model, nn_stack, nn, support, path,
                              no_residuals=False, balanced=False,
                              class_weight=None, test_sfadf=None):
    """
    Generator of S, A, RES arrays from SARS datasets saved in path.

    Args
        model: residual model M: F -> D
        nn_stack (NNStack)
        nn (ConvNet or GenericEncoder)
        support (np.array): support mask for nn
        path (str): path to folder containing 'sars_*.pkl' files (as collected
            with collect_sars_to_disk)
        no_residuals (bool, False): whether to return residuals or dynamics in 
            the RES column of the sares dataset.
        balanced (bool, False): passed to the get_sample_weight method 
        class_weigth (dict, None): passed to the get_sample_weight method 
        test_sfadf (pd.DataFrame, None): compute the test SARES dataset from 
            this dataset.

    Yield
        (S, A, RES, sample_weights, test_sares) (np.array, np.array, np.array,
            np.array, pd.DataFrame): np.arrays with states, actions and 
            residuals, sample weights calculated on the reward column, test 
            SARES dataset calculated using the model fitted on the current SARS,
            for each SARS dataset in path.
    """
    if not path.endswith('/'):
        path += '/'
    files = glob.glob(path + 'sars_*.pkl')
    print 'Got %s files' % len(files)

    for f in files:
        sars = joblib.load(f)
        sfadf = build_sfadf(nn_stack, nn, support, sars)
        F = pds_to_npa(sfadf.F)  # All features from NN stack
        D = pds_to_npa(sfadf.D)  # Feature dynamics of last NN
        sample_weight = get_sample_weight(sars,
                                          balanced=balanced,
                                          class_weight=class_weight,
                                          round_reward=True)
        model.fit(F, D, sample_weight=sample_weight)
        sares = build_sares(model, sfadf, no_residuals=no_residuals)
        S = pds_to_npa(sares.S)
        A = pds_to_npa(sares.A)
        RES = pds_to_npa(sares.RES)

        if test_sfadf:
            test_sares = build_sares(model, test_sfadf, no_residuals=no_residuals)
        else:
            test_sares = None

        yield S, A, RES, sample_weight, test_sares


def build_global_farf(nn_stack, sars):
    """
    Builds FARF' dataset using SARS' dataset:
        F = NN_stack.s_features(S)
        A = A
        R = R
        F' = NN_stack.s_features(S')
        DONE = DONE
    """
    header = ['F', 'A', 'R', 'FF', 'DONE']
    df = pd.DataFrame(columns=header)
    df['F'] = nn_stack.s_features(pds_to_npa(sars.S)).tolist()
    df['A'] = sars.A
    df['R'] = sars.R
    df['FF'] = nn_stack.s_features(pds_to_npa(sars.SS)).tolist()
    df['DONE'] = sars.DONE
    return df


def build_global_farf_from_disk(nn_stack, path):
    """
    Builds FARF' dataset using all SARS' datasets saved in path:
        F = NN_stack.s_features(S)
        A = A
        R = R
        F' = NN_stack.s_features(S')
        DONE = DONE
    """
    if not path.endswith('/'):
        path += '/'
    files = glob.glob(path + 'sars_*.pkl')
    print 'Got %s files' % len(files)
    farf = pd.DataFrame()
    for f in tqdm(files):
        sars = joblib.load(f)
        farf = farf.append(build_global_farf(nn_stack, sars), ignore_index=True)

    return farf


# DATASET HELPERS
def get_class_weight(sars):
    """
    Returns a dictionary with classes (reward values) as keys and weights as
    values.
    The return value can be passed directly to Keras's class_weight parameter
    in model.fit.

    Args
        sars (pd.DataFrame): a SARS' dataset in pandas format.
    """
    if isinstance(sars, pd.DataFrame):
        R = pds_to_npa(sars.R)
    elif isinstance(sars, pd.Series):
        R = pds_to_npa(sars)
    else:
        R = sars

    classes = np.unique(R)
    y = pds_to_npa(R)
    weights = compute_class_weight('balanced', classes, y)
    return dict(zip(classes, weights))


def get_sample_weight(sars, balanced=False, class_weight=None,
                      round_reward=False):
    """
    Returns a list with the class weight of each sample.
    The return value can be passed directly to Keras's sample_weight parameter
    in model.fit

    Args
        sars (pd.DataFrame or pd.Series): a SARS' dataset in pandas format or a
            pd.Series with rewards.
        balanced (bool, False): override class weights and use scikit-learn's 
            weight method
        class_weight (dict, None): dictionary with classes as key and weights as
            values. If None, the dictionary will be computed using sklearn's
            method.
        round (bool, False): round the rewards to the nearest integer before
            applying the class weights.
    """
    if isinstance(sars, pd.DataFrame):
        R = pds_to_npa(sars.R)
    else:
        R = sars

    if round_reward:
        R = np.round(R)

    if class_weight is None and not balanced:
        class_weight = get_class_weight(R)

    sample_weight = [class_weight[r] for r in R]
    return np.array(sample_weight)


def split_dataset_for_ifs(dataset, features='F', target='R'):
    """
    Splits the dataset into x = features + actions, y = target

    Args
        dataset (pd.DataFrame): a dataset in pandas format.
        features (str, 'F'): key to index the dataset
        target (str, 'R'): key to index the dataset
    """
    f = pds_to_npa(dataset[features])
    a = pds_to_npa(dataset['A']).reshape(-1, 1)  # 1D discreet action
    x = np.concatenate((f, a), axis=1)
    y = pds_to_npa(dataset[target])
    return x, y


def split_dataset_for_rfs(dataset, features='F', next_features='FF', target='R'):
    """
    Splits the dataset into f = features, a = actions, ff = features of next
    states, y = target.

    Args
        dataset (pd.DataFrame): a dataset in pandas format.
        features (str, 'F'): key to index the dataset
        next_features (str, 'FF'): key to index the dataset
        target (str, 'R'): key to index the dataset
    """
    f = pds_to_npa(dataset[features])
    a = pds_to_npa(dataset['A']).reshape(-1, 1)  # 1D discreet action
    ff = pds_to_npa(dataset[next_features])
    y = pds_to_npa(dataset[target])
    return f, a, ff, y


def split_dataset_for_fqi(global_farf):
    """
    Splits the dataset into faft = features + actions + features of next state,
    r = reward

    Args
        global_farf (pd.DataFrame): a dataset in pandas format.
    """
    f = pds_to_npa(global_farf.F)
    a = global_farf.A.as_matrix()
    ff = pds_to_npa(global_farf.FF)
    done = global_farf.DONE.as_matrix()
    r = pds_to_npa(global_farf.R)
    faft = np.column_stack((f, a, ff, done))
    return faft, r


def downsample_farf(farf, period=5):
    farf['bin'] = farf.index / period

    farf = pd.DataFrame()

    farf['F'] = farf['F'][::period].reset_index()['F']
    farf['A'] = None  # TODO how to reduce actions
    farf['R'] = farf.groupby('bin')['F'].sum()
    farf['FF'] = farf['FF'].shift(-period + 1)[:-period + 1:period].reset_index()['FF']
    farf['DONE'] = farf.groupby('bin')['DONE'].sum()

    if len(farf) % period != 0:
        farf = farf[:-1]


def build_features(nn, sars):
    """
    Builds F dataset using SARS' dataset:
        F = NN[i].features(S)
    """
    features = nn.all_features(pds_to_npa(sars.S))
    return features
