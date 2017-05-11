from __future__ import print_function
import numpy as np
from PIL import Image
import pandas as pd


def resize_state(to_resize, new_size=(72, 72)):
    """Resizes every image in to_resize to new_size.
    :param to_resize: a numpy array containing a sequence of greyscale images
        (theano dimension ordering (ch, rows, cols) is assumed)
    :param new_size: the size to which resize the images
    :return: a numpy array with the resized images
    """
    # Iterate over channels (th dimension ordering (ch, rows, cols) is assumed)
    resized = []
    for image in to_resize:
        data = Image.fromarray(image).resize(new_size)
        resized.append(np.asarray(data))
    return np.asarray(resized).squeeze()


def crop_state(to_crop, keep_top=False):
    """Crops every image in to_crop to a square.
    :param to_crop: a numpy array containing a sequence of greyscale images to
        crop along axis 1.
    :param keep_top: crop the images keeping the top part.
    :return: the cropped array
    """
    if keep_top:
        return np.split(to_crop, [to_crop.shape[2]], axis=1)[0]
    else:
        return np.split(to_crop, [to_crop.shape[1] - to_crop.shape[2]], axis=1)[1]


def flat2gen(alist):
    """
    :param alist: a 2d list
    :return: a generator for the flattened list
    """
    for item in alist:
        if isinstance(item, list) or isinstance(item, np.ndarray):
            for subitem in item:
                yield subitem
        else:
            yield item


def flat2list(alist, as_tuple=False, as_set=False):
    """
    :param as_tuple: return a tuple instead of a list
    :param as_set: return a set instead of a list
    :param alist: a 2d list
    :return: a flattened version of the list
    """
    output = [i for i in flat2gen(alist)]
    if as_tuple:
        return tuple(output)
    elif as_set:
        return set(output)
    else:
        return output


def onehot_encode(value, nb_categories):
    """
    :param value: discreet value being encoded.
    :param nb_categories: number of possible discreet values being encoded.
    :return: an array of length nb_categories, such that the value-th element
        equals 1 and all the others 0.
    """
    out = [0] * nb_categories
    out[value] = 1
    return out


def p_load(filename):
    """Loads the numpy object stored as the given filename.

    Args
        filename (str): relative path to numpy file.

    Return
        The loaded object.
    """
    out = np.load(filename)
    return out


def p_dump(obj, filename):
    """Dumps an object to numpy file.

    Args
        obj (Object): the object to dump.
        filename (str): the filename to which save the object.
    """
    np.save(filename, obj)


def pds_to_npa(object_array):
    """
    Converts a pandas series of dtype 'object' to a numpy array
    """
    return np.array([_ for _ in object_array])


def is_stuck(state):
    """
    Returns true if the given state does not change along the 0-th axis.
    """
    equals = True
    for i in range(len(state) - 1):
        equals = equals and (np.sum(state[i] - state[i+1]) == 0)
    return equals


def get_dataset_size(dataset, unit='B'):
    """
    Returns the approximated size of a pandas dataframe in the given unit
    """
    factors = {'B': 1.,
               'KB': 1024.,
               'MB': 1048576.,
               'GB': 1073741824.,
               'TB': 1099511627776.}
    return dataset.memory_usage(index=True, deep=True).sum() / factors[unit]


def get_size(structures, unit='B'):
    """
    Returns the approximated size of all pandas dataframes or np.arrays in the
    given list.
    """
    factors = {'B': 1.,
               'KB': 1024.,
               'MB': 1048576.,
               'GB': 1073741824.,
               'TB': 1099511627776.}
    size = 0.
    for s in structures:
        if isinstance(s, np.ndarray):
            size += s.nbytes
        elif isinstance(s, pd.DataFrame) or isinstance(s, pd.Series):
            size += s.memory_usage(index=True, deep=True).sum()

    return size / factors[unit]
