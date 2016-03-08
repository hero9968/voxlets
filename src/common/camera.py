import numpy as np


class Camera(object):
    '''
    This is a projective camera class.
    '''
    def __init__(self):
        self.K = []
        self.H = []

    def set_intrinsics(self, K):
        self.K = K
        self.inv_K = np.linalg.inv(K)

    def set_extrinsics(self, H):
        '''
        Extrinsics are the location of the camera relative to the world origin
        (This was fixed from being the incorrect way round in Jan 2015)
        '''
        self.H = H
        self.inv_H = np.linalg.inv(H)

    def adjust_intrinsic_scale(self, scale):
        '''
        Changes the scaling, effectivly resixing the output image size
        '''
        self.K[0, 0] *= scale
        self.K[1, 1] *= scale
        self.K[0, 2] *= scale
        self.K[1, 2] *= scale
        self.inv_K = np.linalg.inv(self.K)

    def project_points(self, xyz):
        '''
        Projects nx3 points xyz into the camera image.
        Returns their 2D projected location.
        '''
        assert(xyz.shape[1] == 3)

        to_add = np.zeros((3, 1))
        full_K = np.concatenate((self.K, to_add), axis=1)
        full_mat = np.dot(full_K, self.inv_H)

        temp_trans = self._apply_homo_transformation(xyz, full_mat)

        temp_trans[:, 0] /= temp_trans[:, 2]
        temp_trans[:, 1] /= temp_trans[:, 2]

        return temp_trans  # [:, 0:2]

    def inv_project_points(self, uvd):
        '''
        Throws u,v,d points in pixel coords (and depth)
        out into the real world, based on the transforms provided
        '''
        xyz_at_cam_loc = self.inv_project_points_cam_coords(uvd)

        # transforming points under the extrinsics
        return self._apply_normalised_homo_transform(xyz_at_cam_loc, self.H)

    def inv_project_points_cam_coords(self, uvd):
        '''
        As inv_project_points but doesn't do the homogeneous transformation
        '''
        assert(uvd.shape[1] == 3)
        n_points = uvd.shape[0]

        # creating the camera rays
        uv1 = np.hstack((uvd[:, :2], np.ones((n_points, 1))))
        camera_rays = np.dot(self.inv_K, uv1.T).T

        # forming the xyz points in the camera coordinates
        temp = uvd[:, 2][np.newaxis, :].T
        xyz_at_cam_loc = temp * camera_rays

        return xyz_at_cam_loc

    def inv_transform_normals(self, normals):
        '''
        Transforms normals under the camera extrinsics, such that they end up
        pointing the correct way in world space
        '''
        assert normals.shape[1] == 3
        R = np.linalg.inv(self.inv_H[:3, :3])
        #R = self.H[:3, :3]
        #print np.linalg.inv(self.H[:3, :3])
        #print np.linalg.inv(self.H)[:3, :3]
        return np.dot(R, normals.T).T

    def _apply_normalised_homo_transform(self, xyz, trans):
        '''
        Applies homogeneous transform, and also does the normalising...
        '''
        temp = self._apply_homo_transformation(xyz, trans)
        return temp[:, :3] / temp[:, 3][:, np.newaxis]

    def _apply_transformation(self, xyz, trans):
        '''
        Apply a 3x3 transformation matrix to the vertices
        '''
        to_add = np.zeros((3, 1))
        temp_trans = np.concatenate((trans, to_add), axis=1)
        return np.dot(temp_trans, xyz.T).T

    def _apply_homo_transformation(self, xyz, trans):
        '''
        Apply a 4x4 transformation matrix to the vertices
        '''
        n = xyz.shape[0]
        temp = np.concatenate((xyz, np.ones((n, 1))), axis=1).T
        temp_transformed = trans.dot(temp).T
        return temp_transformed

    def estimate_focal_length(self):
        '''
        Trys to guess the focal length from the intrinsics
        '''
        return self.K[0, 0]
