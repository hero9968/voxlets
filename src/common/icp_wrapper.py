import numpy as np
import subprocess as sp
from sklearn.neighbors import NearestNeighbors
import os
import icp
from time import time
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
import scipy.sparse.csgraph


def numpy_to_str(numpy_array):
    "returns a [a, b, c] representatoin of the array"
    init_str = str(numpy_array.ravel())
    init_str = ' '.join(init_str.split())
    return init_str.replace(' ', ', ').replace('\n', '')


def pmicp_wrapper(fixed, floating, R=None, t=None):
    """
    A wrapper for the pmicp library
    https://github.com/ethz-asl/libpointmatcher/
    """

    # todo - reduce precision for speed
    np.savetxt('/tmp/csv_1.csv', fixed)
    np.savetxt('/tmp/csv_2.csv', floating)

    my_env = os.environ.copy()
    my_env["LD_LIBRARY_PATH"] = "/usr/local/lib:" + \
        my_env["LD_LIBRARY_PATH"]

    # generate the command to run
    cmd = ["/usr/local/bin/pmicp", "--output", "/tmp/temp",
                     "--isTransfoSaved", "true"]

    if R is not None:
        cmd.append("--initRotation")
        cmd.append(numpy_to_str(R))

    if t is not None:
        cmd.append("--initTranslation")
        cmd.append(numpy_to_str(t))

    cmd.append('/tmp/csv_1.csv')
    cmd.append('/tmp/csv_2.csv')

    output = sp.call(cmd, env=my_env)
    T = np.genfromtxt('/tmp/temp_complete_transfo.txt')
    R = T[:3, :3]
    t = T[:3, -1]
    return R, t


def valid_uv(uv, im_shape):
    return np.logical_and.reduce((uv[:, 0] < im_shape[1],
                                  uv[:, 1] < im_shape[0],
                                  uv[:, 0] >= 0,
                                  uv[:, 1] >= 0))


def uvd_to_depth_im(uvd, grid_shape):
    """converts a uvd array to a depth image"""
    depth_im = np.zeros(grid_shape) * np.nan
    uv = np.round(uvd[:, :2]).astype(int)
    to_use = valid_uv(uv, grid_shape)

    depth_im[uv[to_use, 1], uv[to_use, 0]] = uvd[to_use, 2]
    return depth_im


def _error_mapping(d):
    '''map a depth difference to an error score
    negative is bad, indicates inconsistancy'''
    d = d.copy()[~np.isnan(d)]
    d[d > 0.01] = 0.01
    d[d < -0.05] = 0.05
    d = np.abs(d)
    return d


def _error_score(d):
    return np.mean(np.abs(_error_mapping(d)))


def _asymmetric_error_metric(im1, im2, mask1, mask2, R, t):
    '''
    given two depth images, each with a point mask, this function
    computes a scalar error metric. The idea is to make use of
    visibility/occlusion in the error metric computation
    '''
    # xyz2 is the floating one, typically
    xyz2 = im2.get_world_xyz()[mask2.ravel()]

    xyz_2_in_pc1_world_space = icp._transform(xyz2, R, t)
    xyz_2_in_cam_1 = im1.cam.project_points(xyz_2_in_pc1_world_space)

    uv = np.round(xyz_2_in_cam_1[:, :2]).astype(int)
    d = xyz_2_in_cam_1[:, 2]

    # compute the difference - but need to make this work for all sorts of errors
    to_use = valid_uv(uv, im1.depth.shape)
    try:
        error = d[to_use] - im1.depth[uv[to_use, 1], uv[to_use, 0]]
    except:
        import pdb; pdb.set_trace()

    return _error_score(error)


def error_metric(im1, im2, mask1, mask2, R, t):
    '''
    final symmetric error metric for two depth images with masks and transforms
    TODO: should I normalise by number of points or something?
    '''
    e1 = _asymmetric_error_metric(im1, im2, mask1, mask2, R, t)

    R_inv = np.linalg.inv(R)
    t_inv = -R_inv.dot(t)
    e2 = _asymmetric_error_metric(im2, im1, mask2, mask1, R_inv, t_inv)
    # print e1, e2
    return e1 + e2


def reproject_points(xyz, cam):
    '''
    returns a matrix uvd of the position of and depth to each xyz point
    when the points have been reprojected into cam
    '''
    return cam.project_points(xyz)


def pairwise_match(fixed, floating, outlier_dist, start_rotations=12):
    '''
    does icp for multiple starting rotaions about the z axis
    returns each possible match
    '''
    # nn = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(fixed)

    possible_transforms = []

    for rot in np.linspace(0, 360, start_rotations):

        # generate an initial rotation matrix about the z axis
        c, s = np.cos(np.deg2rad(rot)), np.sin(np.deg2rad(rot))
        R_init = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

        # work out a good starting translation
        temp_float = icp._transform(floating, R=R_init)
        t_init = fixed.mean(0) - temp_float.mean(0)

        # run the icp and save the result
        R, t = pmicp_wrapper(fixed, floating, R=R_init, t=t_init)

        # form some estimation of the error
        # temp_float = icp._transform(floating, R, t)
        # distances, indices = nn.kneighbors(temp_float)
        # to_use = distances.flatten() < outlier_dist

        # unmatched_floating = np.sum(~to_use)
        # unmatched_fixed = fixed.shape[0] - np.unique(indices.ravel()[to_use]).shape[0]

        # error = icp._get_disparity(
        #     fixed[indices.ravel()[to_use]], temp_float[to_use]) + \
        #     unmatched_fixed * 0.001 + unmatched_floating * 0.005

        # save the result
        possible_transforms.append({'R':R, 't':t})

    return possible_transforms


def best_transform(fixed_region, floating_region, outlier_dist, resample=None):
    '''
    given two depth images, each with masks saying which points to use,
    this function finds the best trasnformation between them.
    This makes the z-axis assumption for initialisation, as in pairwise_match
    '''

    fixed_xyz = fixed_region.get_world_xyz()
    floating_xyz = floating_region.get_world_xyz()

    if resample is not None:
        fixed_xyz = _choose_rows(fixed_xyz, resample)
        floating_xyz = _choose_rows(floating_xyz, resample)

    tic = time()
    possible_transforms = pairwise_match(fixed_xyz, floating_xyz, outlier_dist)
    print "Time is ", time() - tic

    for p in possible_transforms:

        p['error'] = error_metric(
            fixed_region.im, floating_region.im,
            fixed_region.mask, floating_region.mask,
            p['R'], p['t'])
        if np.isnan(p['error']):
            p['error'] = np.inf

    best_idx = np.argmin(np.array([p['error'] for p in possible_transforms]))

    return possible_transforms[best_idx]


def _choose_rows(xyz, number):
    if number >= xyz.shape[0]:
        return xyz
    else:
        idxs = np.random.choice(xyz.shape[0], number, replace=False)
        return xyz[idxs]


# def view_transform(im1, im2, R, t):
'''
this is a function to project im2 into the camera of im1 and create a depth
image, basically just for the purpose of visualisation
'''


class Network(object):
    """
    Undirected graph structure.
    Each vertex is a view
    Each edge stores a vertex and a trasnformation
    (I will make it so that each edge stores the one with the best, i.e. minium error from
    the pairwise transforms)
    """
    def __init__(self):
        self.M = None # M is square matrix perhaps
        self.edge_attributes = {}
        self.n_vertices = 0

    def populate(self, regions, outlier_dist=0.01, resample=None):
        """
        note: transform always should go from lowest to highest
        """

        self.n_vertices = len(regions)
        self.M = np.zeros((self.n_vertices, self.n_vertices))

        for i in range(len(regions)):
            for j in range(i+1, len(regions)):

                best_transformij = best_transform(
                    regions[i], regions[j], outlier_dist, resample)

                best_transformji = best_transform(
                    regions[j], regions[i], outlier_dist, resample)


                if best_transformij['error'] < best_transformji['error']:
                    best_transformji['R'] = np.linalg.inv(best_transformij['R'])
                    best_transformji['t'] = -best_transformji['R'].dot(best_transformij['t'])
                elif best_transformji['error'] < best_transformij['error']:
                    best_transformij['R'] = np.linalg.inv(best_transformji['R'])
                    best_transformij['t'] = -best_transformij['R'].dot(best_transformji['t'])
                #     best_t = best_transformji
                # else:
                #     best_t = best_transformij
                #     best_t['R'] = np.linalg.inv(best_t['R'])
                #     best_t['t'] = - best_t['R'].dot(best_t['t'])
                self.M[i, j] = best_transformij['error']
                self.M[j, i] = best_transformji['error']

                self.edge_attributes[(i, j)] = best_transformij
                self.edge_attributes[(j, i)] = best_transformji

        # ok this is really horrible. sorry future me
        self.M = self.M + self.M.T

    def sort(self, i, j):
        if i > j:
            return (j, i)
        else:
            return (i, j)

    def find_spanning_route(self, i, j):
        """
        finds the chain of edges that links i to j, using the precomputed
        dijkstra adjaciency matrix
        """
        self.dijk = scipy.sparse.csgraph.dijkstra(self.M, return_predecessors=True)[1]

        edge_chain = []
        prev_node = i
        it = j

        while self.dijk[i, it] != i:
            edge_chain.append((prev_node, self.dijk[i, it]))
            it = prev_node = self.dijk[i, it]

        edge_chain.append((prev_node, j))
        return edge_chain
        # scipy.sparse.csgraph.dijkstra(self.M, indices=(3, 4))

    def find_transformation_route(self, i, j):
        # find the transformation which relates i to j
        # todo - should store all transforms as 4x4! makes transformations here much easier...
        edges = self.find_spanning_route(i, j)
        T = np.eye(4)
        for ei, ej in edges:
            tempT = self.full_transform(
                self.get_transform(ei, ej)['R'],
                self.get_transform(ei, ej)['t'])
            # T = T.dot(np.linalg.inv(tempT))
            T = T.dot(tempT)
        return T

    def full_transform(self, R, t):
        full = np.eye(4)
        full[:3, :3] = R
        full[:3, -1] = t
        return full

    def get_transform(self, i, j):
        """returns the transform between two (adjacient?) nodes
        should be able to chain edges together...
        """
        return self.edge_attributes[i, j]


