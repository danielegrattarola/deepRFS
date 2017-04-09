from deep_ifs.utils.helpers import *
from joblib import Parallel, delayed
from tqdm import tqdm
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures


def episode(mdp, policy, video=False, initial_actions=None):
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
        # TODO why the f**k does this happen?
        if not isinstance(action, int):
            action = action[0]
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
                 debug=False, initial_actions=None):
    assert episodes > 0
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

    np.random.shuffle(dataset)

    # TODO debug
    if debug:
        dataset = dataset[:10]

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
    classes = sars.R.unique()
    y = sars.R.as_matrix()
    weights = compute_class_weight('balanced', classes, y)
    return dict(zip(classes, weights))


def get_sample_weight(sars):
    """
    Returns a list with the class weight of each sample.
    The return value can be passed directly to Keras's sample_weight parameter
    in model.fit

    Args
        sars (pd.DataFrame): a SARS' dataset in pandas format.
    """
    class_weight = get_class_weight(sars)
    sample_weight = [class_weight[r] for r in sars.R]
    return np.array(sample_weight)


def split_dataset_for_ifs(dataset, features='F', target='R'):
    f = pds_to_npa(dataset[features])
    a = pds_to_npa(dataset['A']).reshape(-1, 1)  # 1D discreet action
    x = np.concatenate((f, a), axis=1)
    y = pds_to_npa(dataset[target])
    return x, y


def split_dataset_for_rfs(dataset, features='F', next_features='FF', target='R'):
    f = pds_to_npa(dataset[features])
    a = pds_to_npa(dataset['A']).reshape(-1, 1)  # 1D discreet action
    ff = pds_to_npa(dataset[next_features])
    y = pds_to_npa(dataset[target])
    return f, a, ff, y


def split_dataset_for_fqi(global_farf):
    f = pds_to_npa(global_farf.F)
    a = global_farf.A.as_matrix()
    ff = pds_to_npa(global_farf.FF)
    done = global_farf.DONE.as_matrix()
    r = pds_to_npa(global_farf.R)
    faft = np.column_stack((f,a,ff,done))
    return faft, r


def build_farf(nn, sars):
    """
    Builds FARF' dataset using SARS' dataset:
        F = NN[0].features(S)
        A = A
        R = R
        F' = NN[0].features(S')
    """
    farf = []
    for datapoint in sars.itertuples():
        f = nn.all_features(np.expand_dims(datapoint.S, 0))
        a = datapoint.A
        r = datapoint.R
        ff = nn.all_features(np.expand_dims(datapoint.SS, 0))
        farf.append([f, a, r, ff])
    farf = np.array(farf)
    header = ['F', 'A', 'R', 'FF']
    return pd.DataFrame(farf, columns=header)


def build_sfadf(nn_stack, nn, support, sars):
    """
    # Builds SFADF' dataset using SARS' dataset:
        S = S
        F = NN_stack.s_features(S)
        A = A
        D = NN[i-1].s_features(S) - NN[i-1].s_features(S')
        F' = NN_stack.s_features(S')
    """
    sfadf = []
    for datapoint in sars.itertuples():
        s = datapoint.S
        f = nn_stack.s_features(np.expand_dims(datapoint.S, 0))
        a = datapoint.A
        d = nn.s_features(np.expand_dims(datapoint.S, 0), support) - \
            nn.s_features(np.expand_dims(datapoint.SS, 0), support)
        ff = nn_stack.s_features(np.expand_dims(datapoint.SS, 0))
        sfadf.append([s, f, a, d, ff])
    sfadf = np.array(sfadf)
    header = ['S', 'F', 'A', 'D', 'FF']
    return pd.DataFrame(sfadf, columns=header)


def build_sares(model, sfadf):
    """
    Builds SARes dataset from SFADF':
        S = S
        A = A
        Res = D - M(F)
    """
    sares = []
    for datapoint in sfadf.itertuples():
        s = datapoint.S
        a = datapoint.A

        features = np.expand_dims(datapoint.F, 0)
        # if model_type == 'linear':
        #     features = PolynomialFeatures(degree=5).fit_transform(features)

        prediction = model.predict(features)
        res = datapoint.D - prediction
        sares.append([s, a, res])
    sares = np.array(sares)
    header = ['S', 'A', 'RES']
    return pd.DataFrame(sares, columns=header)


def build_fadf(nn_stack, nn, sars, sfadf):
    """
    Builds new FADF' dataset from SARS' and SFADF':
        F = NN_stack.s_features(S) + NN[i].features(S)
        A = A
        D = SFADF'.D
        F' = NN_stack.s_features(S') + NN[i].features(S')
    """
    faf = []
    for datapoint in sars.itertuples():
        f = np.append(nn_stack.s_features(np.expand_dims(datapoint.S, 0)),
                      nn.all_features(np.expand_dims(datapoint.S, 0)))
        a = datapoint.A
        ff = np.append(nn_stack.s_features(np.expand_dims(datapoint.SS, 0)),
                       nn.all_features(np.expand_dims(datapoint.SS, 0)))
        faf.append([f, a, ff])
    faf = np.array(faf)
    header = ['F', 'A', 'FF']
    fadf = pd.DataFrame(faf, columns=header)
    fadf['D'] = sfadf.D
    fadf = fadf[['F', 'A', 'D', 'FF']]
    return fadf

def build_fadf_no_preload(nn, sars, sfadf):
    """
    Builds new FADF' dataset from SARS' and SFADF':
        F = NN_stack.s_features(S) + NN[i].features(S)
        A = A
        D = SFADF'.D
        F' = NN_stack.s_features(S') + NN[i].features(S')
    """
    faf = []
    for datapoint in sars.itertuples():
        f = nn.all_features(np.expand_dims(datapoint.S, 0))
        a = datapoint.A
        ff = nn.all_features(np.expand_dims(datapoint.SS, 0))
        faf.append([f, a, ff])
    faf = np.array(faf)
    header = ['F', 'A', 'FF']
    fadf = pd.DataFrame(faf, columns=header)
    fadf['D'] = sfadf.D
    fadf = fadf[['F', 'A', 'D', 'FF']]
    return fadf


def build_global_farf(nn_stack, sars):
    """
    Builds FARF' dataset using SARS' dataset:
        F = NN_stack.s_features(S)
        A = A
        R = R
        F' = NN_stack.s_features(S')
    """
    farf = []
    for datapoint in sars.itertuples():
        f = nn_stack.s_features(np.expand_dims(datapoint.S, 0))
        a = datapoint.A
        r = datapoint.R
        ff = nn_stack.s_features(np.expand_dims(datapoint.SS, 0))
        done = datapoint.DONE
        farf.append([f, a, r, ff, done])
    farf = np.array(farf)
    header = ['F', 'A', 'R', 'FF', 'DONE']
    return pd.DataFrame(farf, columns=header)
