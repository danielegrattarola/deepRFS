import numpy as np
import pandas as pd
from deep_ifs.utils.helpers import flat2list, pds_to_npa, is_stuck
from joblib import Parallel, delayed
from tqdm import tqdm
from sklearn.utils.class_weight import compute_class_weight


def episode(mdp, policy, video=False, initial_actions=None):
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
    patience = mdp.action_space.n

    # Get current state
    state = mdp.reset()

    # Force start
    if initial_actions is not None:
        state, _, _, _ = mdp.step(np.random.choice(initial_actions))

    reward = 0
    done = False

    # Start episode
    ep_output = []
    while not done:
        frame_counter += 1

        # Select and execute the action, get next state and reward
        action = policy.draw_action(np.expand_dims(state, 0), done)
        action = int(action)
        next_state, reward, done, info = mdp.step(action)

        if is_stuck(next_state):
            patience -= 1
        if patience == 0:
            patience = mdp.action_space.n
            next_state, reward, done, info = mdp.step(1)  # Force start

        # build SARS' tuple
        ep_output.append([state, action, reward, next_state, done])

        # Render environment
        if video:
            mdp.render()

        # Update state
        state = next_state

    return ep_output


def collect_sars(mdp, policy, episodes=100, n_jobs=1, random_greedy_split=0.9,
                 debug=False, initial_actions=None, shuffle=True):
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
        delayed(episode)(mdp, policy) for _ in tqdm(xrange(random_episodes))
    )
    # Each episode is in a list, so the dataset needs to be flattened
    dataset_random = np.asarray(flat2list(dataset_random))

    policy.set_epsilon(0)
    dataset_greedy = Parallel(n_jobs=n_jobs)(
        delayed(episode)(mdp, policy, initial_actions=initial_actions)
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


def get_sample_weight(sars, class_weight=None, round=False):
    """
    Returns a list with the class weight of each sample.
    The return value can be passed directly to Keras's sample_weight parameter
    in model.fit

    Args
        sars (pd.DataFrame or pd.Series): a SARS' dataset in pandas format or a
            pd.Series with rewards.
    """
    if isinstance(sars, pd.DataFrame):
        R = pds_to_npa(sars.R)
    else:
        R = sars

    if round:
        R = np.round(R)

    if class_weight is None:
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
    # Builds SFADF' dataset using SARS' dataset:
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
    # Builds SFAD dataset using SARS' dataset:
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


def build_sares(model, sfadf):
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
    predictions = model.predict(features)
    residuals = dynamics - predictions
    df['RES'] = residuals.tolist()
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


def build_global_farf(nn_stack, sars):
    """
    Builds FARF' dataset using SARS' dataset:
        F = NN_stack.s_features(S)
        A = A
        R = R
        F' = NN_stack.s_features(S')
    """
    header = ['F', 'A', 'R', 'FF', 'DONE']
    df = pd.DataFrame(columns=header)
    df['F'] = nn_stack.s_features(pds_to_npa(sars.S)).tolist()
    df['A'] = sars.A
    df['R'] = sars.R
    df['FF'] = nn_stack.s_features(pds_to_npa(sars.SS)).tolist()
    df['DONE'] = sars.DONE
    return df


def build_features(nn, sars):
    """
    Builds F dataset using SARS' dataset:
        F = NN[i].features(S)
    """
    features = nn.all_features(pds_to_npa(sars.S))
    return features
