import numpy as np

def euclidean_distance(x,Y):
    """
    Compute the Euclidean distance between a vector x and a matrix Y.

    Parameters:
    x (numpy.ndarray): A 1D array of shape (d,).
    Y (numpy.ndarray): A 2D array of shape (n, d).

    Returns:
    numpy.ndarray: A 1D array of shape (n,) containing the Euclidean distances.
    """
    return np.sqrt(np.sum((Y - x) ** 2, axis=1))

def manhattan_distance(x,Y):
    """
    Compute the Manhattan distance between a vector x and a matrix Y.

    Parameters:
    x (numpy.ndarray): A 1D array of shape (d,).
    Y (numpy.ndarray): A 2D array of shape (n, d).

    Returns:
    numpy.ndarray: A 1D array of shape (n,) containing the Manhattan distances.
    """
    return np.sum(np.abs(Y - x), axis=1)
