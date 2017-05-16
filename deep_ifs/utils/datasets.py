import glob
import os

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm

from deep_ifs.extraction.ConvNet import ConvNet
from deep_ifs.utils.helpers import flat2list, pds_to_npa
from deep_ifs.utils.timer import log


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
        state, _, _, info = mdp.step(action)
        lives_count = info['ale.lives']

    reward = 0
    done = False

    # Start episode
    ep_output = []
    while not done:
        frame_counter += 1

        if initial_actions is not None:
            if info['ale.lives'] < lives_count:
                lives_count = info['ale.lives']
                state, _, _, _ = mdp.step(np.random.choice(initial_actions))

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
                 debug=False, initial_actions=None, shuffle=True, repeat=1,
                 return_dataframe=False):
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
        dataset[:2, 2] = 1.0
        dataset[6, 2] = -1.0

    if return_dataframe:
        header = ['S', 'A', 'R', 'SS', 'DONE']
        return pd.DataFrame(dataset, columns=header)
    else:
        return dataset


def collect_sars_to_disk(mdp, policy, path, datasets=1, episodes=100,
                         n_jobs=1, random_greedy_split=0.9, debug=False,
                         initial_actions=None, shuffle=True, repeat=1,
                         batch_size=None):
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

    samples_in_dataset = 0
    for i in range(datasets):
        sars = collect_sars(mdp, policy, episodes=episodes, n_jobs=n_jobs,
                            random_greedy_split=random_greedy_split,
                            debug=debug, initial_actions=initial_actions,
                            shuffle=shuffle, repeat=repeat)

        # Cut dataset to match batch size
        if batch_size is not None:
            excess = len(sars) % batch_size
            if excess > 0:
                sars = sars[:-excess]

        log('Got %s samples (dropped %s)' % (len(sars), excess))
        samples_in_dataset += len(sars)
        np.save(path + 'sars_%s.npy' % i, sars)

    return samples_in_dataset


def sar_generator_from_disk(path, batch_size=32, balanced=False,
                            class_weight=None, binarize=False, clip=False):
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
    files = glob.glob(path + 'sars_*.npy')
    print 'Got %s files' % len(files)

    # If balanced, compute class weight on all rewards
    if class_weight is None or balanced:
        class_weight = get_class_weight_from_disk(path, clip_target=clip)

    while True:
        for idx, f in enumerate(files):
            sars = np.load(f)
            if clip:
                sars[:, 2] = np.clip(sars[:, 2], -1, 1)
            if idx > 0:
                sars = np.append(excess_sars, sars, axis=0)

            excess = len(sars) % batch_size
            if excess > 0:
                excess_sars = sars[-excess:]
                sars = sars[:-excess]
            else:
                excess_sars = sars[0:0]  # just to preserve shapes

            nb_batches = len(sars) / batch_size

            sample_weight = get_sample_weight(pds_to_npa(sars[:, 2]),
                                              balanced=False,  # Dealt with manually
                                              class_weight=class_weight)

            for i in range(nb_batches):
                start = i * batch_size
                stop = (i + 1) * batch_size
                S = pds_to_npa(sars[start:stop, 0])
                A = pds_to_npa(sars[start:stop, 1])
                R = pds_to_npa(sars[start:stop, 2])

                # Preprocess data
                S = ConvNet.preprocess_state(S, binarize=binarize)

                yield ([S, A], R, sample_weight[start:stop])


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


def build_f_from_disk(nn, path):
    """
    Builds F dataset using SARS' dataset:
        F = NN[i].features(S)
    """
    if not path.endswith('/'):
        path += '/'
    files = glob.glob(path + 'sars_*.npy')
    print 'Got %s files' % len(files)

    for idx, f in enumerate(files):
        sars = np.load(f)
        if idx == 0:
            F = nn.all_features(pds_to_npa(sars[:, 0]))
        else:
            new_F = nn.all_features(pds_to_npa(sars[:, 0]))
            F = np.append(F, new_F, axis=0)

    return F


def build_far_from_disk(nn, path, clip=False):
    if not path.endswith('/'):
        path += '/'
    files = glob.glob(path + 'sars_*.npy')
    print 'Got %s files' % len(files)

    for idx, f in enumerate(files):
        sars = np.load(f)
        if clip:
            sars[:, 2] = np.clip(sars[:, 2], -1, 1)
        if idx == 0:
            F = nn.all_features(pds_to_npa(sars[:, 0]))
            A = pds_to_npa(sars[:, 1])
            R = pds_to_npa(sars[:, 2])
        else:
            new_F = nn.all_features(pds_to_npa(sars[:, 0]))
            new_A = pds_to_npa(sars[:, 1])
            new_R = pds_to_npa(sars[:, 2])
            F = np.append(F, new_F, axis=0)
            A = np.append(A, new_A, axis=0)
            R = np.append(R, new_R, axis=0)

    A = A.reshape(-1, 1)
    FA = np.concatenate((F, A), axis=1)

    # Post processing
    R = R.reshape(-1, 1)  # Sklearn version < 0.19 will throw a warning
    return FA, R


def build_fd(nn_stack, nn, support, sars):
    F = nn_stack.s_features(pds_to_npa(sars[:, 0]))
    D = nn.s_features(pds_to_npa(sars[:, 0]), support) - \
        nn.s_features(pds_to_npa(sars[:, 3]), support)
    return F, D


def build_fd_from_disk(nn_stack, nn, support, path):
    if not path.endswith('/'):
        path += '/'
    files = glob.glob(path + 'sars_*.npy')
    print 'Got %s files' % len(files)

    for idx, f in enumerate(files):
        sars = np.load(f)
        if idx == 0:
            F = nn_stack.s_features(pds_to_npa(sars[:, 0]))
            D = nn.s_features(pds_to_npa(sars[:, 0]), support) - \
                nn.s_features(pds_to_npa(sars[:, 3]), support)
        else:
            new_F = nn_stack.s_features(pds_to_npa(sars[:, 0]))
            new_D = nn.s_features(pds_to_npa(sars[:, 0]), support) - \
                    nn.s_features(pds_to_npa(sars[:, 3]), support)
            F = np.append(F, new_F, axis=0)
            D = np.append(D, new_D, axis=0)

    return F, D


# IFS i
def build_fa_from_disk(nn_stack, nn, path):
    if not path.endswith('/'):
        path += '/'
    files = glob.glob(path + 'sars_*.npy')
    print 'Got %s files' % len(files)

    for idx, f in enumerate(files):
        sars = np.load(f)
        if idx == 0:
            F = np.column_stack((nn_stack.s_features(pds_to_npa(sars[:, 0])),
                                nn.all_features(pds_to_npa(sars[:, 0]))))
            A = pds_to_npa(sars[:, 1])
        else:
            new_F = np.column_stack((nn_stack.s_features(pds_to_npa(sars[:, 0])),
                                    nn.all_features(pds_to_npa(sars[:, 0]))))
            new_A = pds_to_npa(sars[:, 1])
            F = np.append(F, new_F, axis=0)
            A = np.append(A, new_A, axis=0)

    A = A.reshape(-1, 1)
    FA = np.concatenate((F, A), axis=1)
    return FA


def build_res(model, F, D, no_residuals=False):
    if no_residuals:
        RES = D
    else:
        predictions = model.predict(F)
        RES = D - predictions
    return RES


# NNi
def sares_generator_from_disk(model, nn_stack, nn, support, path, batch_size=32,
                              binarize=False, no_residuals=False,
                              use_sample_weights=True, balanced=False,
                              class_weight=None):
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
    """
    if not path.endswith('/'):
        path += '/'
    files = glob.glob(path + 'sars_*.npy')
    print 'Got %s files' % len(files)

    # If balanced, compute class weight on all rewards
    if use_sample_weights and (class_weight is None or balanced):
        class_weight = get_class_weight_from_disk(path)

    while True:
        for idx, f in enumerate(files):
            sars = np.load(f)
            if idx > 0:
                sars = np.append(excess_sars, sars, axis=0)

            excess = len(sars) % batch_size
            if excess > 0:
                excess_sars = sars[-excess:]
                sars = sars[:-excess]
            else:
                excess_sars = sars[0:0]  # just to preserve shapes

            nb_batches = len(sars) / batch_size

            # Compute sample_weights over reward
            if use_sample_weights:
                sample_weight = get_sample_weight(sars[:, 2],
                                                  balanced=False,  # Dealt with manually
                                                  class_weight=class_weight)

            for i in range(nb_batches):
                start = i * batch_size
                stop = (i + 1) * batch_size
                S = pds_to_npa(sars[start:stop, 0])
                A = pds_to_npa(sars[start:stop, 1])
                F, D = build_fd(nn_stack, nn, support, sars[start:stop])
                RES = build_res(model, F, D, no_residuals=no_residuals)

                # Preprocess data
                S = ConvNet.preprocess_state(S, binarize=binarize)

                if use_sample_weights:
                    yield ([S, A], RES, sample_weight[start:stop])
                else:
                    yield ([S, A], RES)


def build_fart_r_from_disk(nn_stack, path):
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
    files = glob.glob(path + 'sars_*.npy')
    print 'Got %s files' % len(files)

    for idx, f in enumerate(files):
        sars = np.load(f)
        if idx == 0:
            F = nn_stack.s_features(pds_to_npa(sars[:, 0]))
            A = pds_to_npa(sars[:, 1])
            R = pds_to_npa(sars[:, 2])
            FF = nn_stack.s_features(pds_to_npa(sars[:, 3]))
            DONE = pds_to_npa(sars[:, 4])
        else:
            new_F = nn_stack.s_features(pds_to_npa(sars[:, 0]))
            new_A = pds_to_npa(sars[:, 1])
            new_R = pds_to_npa(sars[:, 2])
            new_FF = nn_stack.s_features(pds_to_npa(sars[:, 3]))
            new_DONE = pds_to_npa(sars[:, 4])
            F = np.append(F, new_F, axis=0)
            A = np.append(A, new_A, axis=0)
            R = np.append(R, new_R, axis=0)
            FF = np.append(FF, new_FF, axis=0)
            DONE = np.append(DONE, new_DONE, axis=0)

    faft = np.column_stack((F, A, FF, DONE))
    action_values = np.unique(A)
    return faft, R, action_values


# DATASET HELPERS
def get_class_weight(target):
    """
    Returns a dictionary with classes (reward values) as keys and weights as
    values.
    The return value can be passed directly to Keras's class_weight parameter
    in model.fit.

    Args
        sars (pd.DataFrame): a SARS' dataset in pandas format.
    """
    if isinstance(target, pd.DataFrame):
        target = pds_to_npa(target.R)
    else:
        target = pds_to_npa(target)

    classes = np.unique(target)
    weights = compute_class_weight('balanced', classes, target)
    return dict(zip(classes, weights))


def get_class_weight_from_disk(path, clip_target=False):
    """
    Returns a list with the class weight of each sample.
    The return value can be passed directly to Keras's sample_weight parameter
    in model.fit

    Args
        target (pd.DataFrame or pd.Series): a SARS' dataset in pandas format or a
            pd.Series with rewards.
        balanced (bool, False): override class weights and use scikit-learn's 
            weight method
        class_weight (dict, None): dictionary with classes as key and weights as
            values. If None, the dictionary will be computed using sklearn's
            method.
        round (bool, False): round the rewards to the nearest integer before
            applying the class weights.
    """
    if not path.endswith('/'):
        path += '/'
    files = glob.glob(path + 'sars_*.npy')
    print 'Got %s files' % len(files)
    for idx, f in enumerate(files):
        sars = np.load(f)
        if idx == 0:
            target = pds_to_npa(sars[:, 2])
        else:
            target = np.append(target, pds_to_npa(sars[:, 2]))

    if clip_target:
        target = np.clip(target, -1, 1)

    classes = np.unique(target)
    weights = compute_class_weight('balanced', classes, target)
    return dict(zip(classes, weights))


def get_sample_weight(target, balanced=False, class_weight=None):
    """
    Returns a list with the class weight of each sample.
    The return value can be passed directly to Keras's sample_weight parameter
    in model.fit

    Args
        target (pd.DataFrame or pd.Series): a SARS' dataset in pandas format or a
            pd.Series with rewards.
        balanced (bool, False): override class weights and use scikit-learn's 
            weight method
        class_weight (dict, None): dictionary with classes as key and weights as
            values. If None, the dictionary will be computed using sklearn's
            method.
        round (bool, False): round the rewards to the nearest integer before
            applying the class weights.
    """
    if isinstance(target, pd.DataFrame):
        target = pds_to_npa(target.R)
    else:
        target = pds_to_npa(target)

    if class_weight is None or balanced:
        class_weight = get_class_weight(target)

    sample_weight = [class_weight[r] for r in target]
    return np.array(sample_weight)


def get_nb_samples_from_disk(path):
    if not path.endswith('/'):
        path += '/'
    files = glob.glob(path + 'sars_*.npy')
    print 'Got %s files' % len(files)

    result = 0
    for f in files:
        sars = np.load(f)
        result += len(sars)

    return result