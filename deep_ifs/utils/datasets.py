from deep_ifs.utils.helpers import *
from joblib import Parallel, delayed
from tqdm import tqdm
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures


def episode(env, policy, video=False):
    frame_counter = 0

    # Get current state
    state = env.reset()
    reward = 0
    done = False

    # Start episode
    ep_output = []
    while not done:
        frame_counter += 1

        # Select and execute the action, get next state and reward
        action = policy.draw_action(np.expand_dims(state, 0), done)
        next_state, reward, done, info = env.step(action)

        # build SARS' tuple
        ep_output.append([state, action, reward, next_state, done])

        # Render environment
        if video:
            env.render()

        # Update state
        state = next_state

    return ep_output


def collect_sars(env, policy, episodes=100, n_jobs=1, debug=False):
    # Collect episodes in parallel
    dataset = Parallel(n_jobs=n_jobs)(
        delayed(episode)(env, policy) for _ in tqdm(xrange(episodes))
    )
    # Each episode is in a list, so the dataset needs to be flattened
    dataset = np.asarray(flat2list(dataset))

    # TODO debug
    if debug:
        dataset = dataset[:10]

    header = ['S', 'A', 'R', 'SS', 'DONE']
    return pd.DataFrame(dataset, columns=header)


def get_class_weight(sars):
    """
    Takes as input a SARS' dataset in pandas format.
    Returns a dictionary with classes (reward values) as keys and weights as
    values.
    The return value can be passed directly to Keras's class_weight parameter
    in model.fit
    """
    classes = sars.R.unique()
    y = sars.R.as_matrix()
    weights = compute_class_weight('balanced', classes, y)
    return dict(zip(classes, weights))


def get_sample_weight(sars):
    """
    Takes as input a SARS' dataset in pandas format.
    Returns a list with the class weight of each sample.
    The return value can be passed directly to Keras's sample_weight parameter
    in model.fit
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


def split_dataset_for_fqi(global_farf):
    f = pds_to_npa(global_farf.F)
    a = global_farf.A.as_matrix()
    ff = pds_to_npa(global_farf.FF)
    done = global_farf.DONE.as_matrix()
    r = pds_to_npa(global_farf.R)
    faft = np.column_stack((f,a,ff,done))
    return faft, r


def build_farf(nn, sars):
    # Build FARF' dataset using SARS' dataset:
    # F = NN[0].features(S)
    # A = A
    # R = R
    # F' = NN[0].features(S')
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
    # Build SFADF' dataset using SARS' dataset:
    # S = S
    # F = NN_stack.s_features(S)
    # A = A
    # D = NN[i-1].s_features(S) - NN[i-1].s_features(S')
    # F' = NN_stack.s_features(S')
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


def build_sares(model, sfadf, model_type='linear'):
    # Build SARes dataset from SFADF':
    # S = S
    # A = A
    # Res = D - M(F)
    sares = []
    for datapoint in sfadf.itertuples():
        s = datapoint.S
        a = datapoint.A

        features = np.expand_dims(datapoint.F, 0)
        if model_type == 'linear':
            features = PolynomialFeatures(degree=5).fit_transform(features)

        prediction = model.predict(features)
        res = datapoint.D - prediction
        sares.append([s, a, res])
    sares = np.array(sares)
    header = ['S', 'A', 'RES']
    return pd.DataFrame(sares, columns=header)


def build_fadf(nn_stack, nn, sars, sfadf):
    # Build new FADF' dataset from SARS' and SFADF':
    # F = NN_stack.s_features(S) + NN[i].features(S)
    # A = A
    # D = SFADF'.D
    # F' = NN_stack.s_features(S') + NN[i].features(S')
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


def build_global_farf(nn_stack, sars):
    # Build FARF' dataset using SARS' dataset:
    # F = NN_stack.s_features(S)
    # A = A
    # R = R
    # F' = NN_stack.s_features(S')
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
