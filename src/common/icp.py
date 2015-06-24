import numpy as np
from sklearn.neighbors import NearestNeighbors


def icp(fixed, floating, R=None, t=None, no_iterations=100, outlier_dist=None,
        constraint=None, transform_type='point_to_plane', normals=None):
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

    MF:
    I called the point clouds fixed and floating because I can never remember
    which is model and which is template.
    The fixed one stays fixed, the floating one is moved close to the fixed one.
    '''
    assert floating.shape[1] == fixed.shape[1], \
        'Error - both clouds must be of the same dimensionality'
    N, d = fixed.shape

    if t is None:
        t = np.zeros(d,)

    if R is None:
        R = np.eye(d)

    # choosing which algorithm to use
    if transform_type=='point_to_plane':
        if normals is None:
            raise ValueError('Must give normals')
        else:
            def transform_f(fixed, norms, floating):
                return point_to_plane_align(fixed, norms, floating)
    elif transform_type=='point_to_point':
        def transform_f(fixed, norms, floating):
            return procrustes(fixed, floating)

    # fit nearest neighbour model to the fixed cloud
    nn = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(fixed)

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

        if outlier_dist:
            to_use = distances.flatten() < outlier_dist

            if to_use.sum() < 3:
                raise Exception("Error - not enough points to match")

            #Compute the transformation between the current source
            #and destination cloudpoint
            temp_fixed = fixed[indices.ravel()[to_use]].copy()
            temp_floating = floating[to_use].copy()
            if normals is not None:
                temp_fixed_norms = normals[indices.ravel()[to_use]]
            else:
                temp_fixed_norms = floating

            if constraint is None:
                # no constraints, full 6DOF transformation
                R, t, error = transform_f(temp_fixed, temp_fixed_norms, temp_floating)

            elif constraint == 'z':

                tR, tt, error = transform_f(
                    temp_fixed[:, :2], temp_fixed_norms[:, :2], temp_floating[:, :2])
                R = np.array([[tR[0, 0], tR[0, 1], 0], [tR[1, 0], tR[1, 1], 0], [0, 0, 1]])
                t = np.array([tt[0], tt[1], 0])
            else:
                raise ValueError('Unknown constraint', constraint)

        else:
            #Compute the transformation between the current source
            #and destination cloudpoint
            R, t, error = transform_f(fixed[indices.ravel()].copy(), None, floating.copy())

        if error == prev_error:
            print "Breaking after %d iterations" % i
            break
        elif error > prev_error:
            print "Error has increased! - after %d iterations" % i
            break

        prev_error = error

    inlier_count = to_use.sum()

    return R, t, error, inlier_count


def point_to_plane_align(fixed, fixed_normals, floating):
    '''
    following: http://www.cs.princeton.edu/~smr/papers/icpstability.pdf
    p is floating
    q is fixed
    '''
    d = fixed.shape[1]

    # computing c
    c = np.cross(floating, fixed_normals, 1)
    print "c is shape", c.shape

    stack = np.hstack((c, fixed_normals))
    print stack.shape

    # compute M
    M = np.zeros(d)
    for s in stack:
        M += np.outer(s, s)

    print "M is "
    print M

    # compute b (hopefully the broadcasting will work here)
    p_minus_q = floating - fixed
    b = c * np.dot(p_minus_q, normals)

    # now solve Ma=b
    # (should really use the cholesky decomposition...)
    a = scipy.linalg.solve(M, b)

    # now reform the rotation matrix etc
    angles, t = np.split(a, 2)
    print "Angles is ", angles

    if d == 2:
        alpha, _ = angles
        R = np.eye(2)
        R[0, 1] = -alpha
        R[1, 0] = alpha

    elif d == 3:
        alpha, beta, gamma = angles
        R = np.eye(3)
        R[0, 1] = -gamma
        R[1, 0] = gamma
        R[0, 2] = -beta
        R[2, 0] = beta
        R[1, 2] = -alpha
        R[2, 1] = alpha

    else:
        raise ValueError('Only can deal with 2 or 3 dims at the moment')


    # orthonormalise the rotation matrix (from libicp)
    u, v, d = np.linalg.svd(R)
    R = np.dot(u, d.T)

    # compute the error
    p_trans = _transform(p, R, t)
    disparity = _get_disparity(q, p_trans)

    return R, t, disparity


def _transform(X, R=np.eye(3), t=np.zeros(3,)):
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
    # this is the svd bit!
    mtx2, R = _match_points(mtx1, mtx2)

    disparity = _get_disparity(mtx1, mtx2)

    # compile t1, t2, and R into a single R and t
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
        outMatrix to scale the data for.

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
    q = np.dot(vh.T, u.T)
    # new_mtx2 *= np.sum(s)

    # don't want any reflections for ICP
    if np.linalg.det(q) < 0:
        vh[:,-1] *= -1
        # s[-1] *= -1  should uncomment this if allowing for scale
        q = np.dot(vh.T, u.T)

    new_mtx2 = np.dot(mtx2, q)
    return new_mtx2, q


def _get_disparity(mtx1, mtx2):
    return(np.sum(np.square(mtx1 - mtx2)))

