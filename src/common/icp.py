import numpy as np
from sklearn.neighbors import NearestNeighbors


def icp(fixed, floating, R=np.eye(2), t=np.array([0, 0]), no_iterations = 13):
    '''
    The Iterative Closest Point estimator.
    Takes two cloudpoints a[x,y], b[x,y], an initial estimation of
    their relative pose and the number of iterations
    Returns the affine transform that transforms
    the cloudpoint a to the cloudpoint b.
    Note:
        (1) This method works for cloudpoints with minor
        transformations. Thus, the result depents greatly on
        the initial pose estimation.
        (2) A large number of iterations does not necessarily
        ensure convergence. Contrarily, most of the time it
        produces worse results.
    '''

    # fit nearest neighbour model to the fixed cloud
    nn = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(fixed)

    trans = [floating.copy()]
    prev_error = np.inf

    for i in range(no_iterations):

        # transform floating point cloud under current transform
        temp_floating = _transform(floating, R, t)

        #Find the nearest neighbours between the current source and the
        #destination cloudpoint
        distances, indices = nn.kneighbors(temp_floating)

        if np.unique(indices).shape[0] == 1:
            print "Error - Everyone is attracted to one point"
            return None, None

        #Compute the transformation between the current source
        #and destination cloudpoint
        R, t, error = procrustes(fixed[indices.ravel()].copy(), floating.copy())

        if error == prev_error:
            print "Breaking after %d iterations" % i
            break
        elif error > prev_error:
            print "Error has increased! - after %d iterations" % i
            break

        prev_error = error

    return R, t

def _transform(X, R, t):
    # print "Xt is ", X.shape, T.shape
    return np.dot(R, X.T).T + t



def procrustes(data1, data2):
    """
    Procrustes analysis, a similarity test for two data sets
    Modification of the routine in skbio
    """
    num_rows, num_cols = np.shape(data1)
    if (num_rows, num_cols) != np.shape(data2):
        raise ValueError("input matrices must be of same shape")
    if num_rows == 0 or num_cols == 0:
        raise ValueError("input matrices must be >0 rows, >0 cols")

    # standardize each matrix
    mtx1 = _center(data1)
    mtx2 = _center(data2)

    t1 = data1.mean(0)
    t2 = data2.mean(0)

    if (not np.any(mtx1)) or (not np.any(mtx2)):
        raise ValueError("input matrices must contain >1 unique points")

    mtx1 = _normalize(mtx1)
    mtx2 = _normalize(mtx2)

    # transform mtx2 to minimize disparity (sum( (mtx1[i,j] - mtx2[i,j])^2) )
    mtx2, R = _match_points(mtx1, mtx2)

    disparity = _get_disparity(mtx1, mtx2)

    inv_R = np.linalg.inv(R)
    t = - inv_R.dot(t2) + t1

    # return np.linalg.inv(R), t, disparity
    return inv_R, t, disparity


def _center(mtx):
    # d means double
    result = np.array(mtx, 'd')
    result -= np.mean(result, 0)
    # subtract each column's mean from each element in that column
    return result


def _normalize(mtx):
    """change scaling of data (in rows) such that trace(mtx*mtx') = 1

    Parameters
    ----------
    mtx : array_like
        Matrix to scale the data for.

    Notes
    -----
    mtx' denotes the transpose of mtx

    """
    mtx = np.asarray(mtx, dtype=float)
    return mtx / np.linalg.norm(mtx)


def _match_points(mtx1, mtx2):
    """Returns a transformed mtx2 that matches mtx1.

    Returns
    -------

    A new matrix which is a transform of mtx2.  Scales and rotates a copy of
    mtx 2.  See procrustes docs for details.

    """
    u, s, vh = np.linalg.svd(np.dot(np.transpose(mtx1), mtx2))
    q = np.dot(np.transpose(vh), np.transpose(u))
    new_mtx2 = np.dot(mtx2, q)
    # new_mtx2 *= np.sum(s)

    return new_mtx2, q


def _get_disparity(mtx1, mtx2):
    return(np.sum(np.square(mtx1 - mtx2)))

