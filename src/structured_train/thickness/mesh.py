import numpy as np
#https://github.com/dranjan/python-plyfile
import paths
import h5py

class Mesh(object):
    '''
    class for storing mesh data eg as read from a ply file
    '''
    #from plyfile import PlyData, PlyElement

    def __init__(self):
        self.vertices = []
        self.faces = []


    def load_from_ply(self, filename):
        '''
        loads faces and vertices from a ply file
        '''
        plydata = PlyData.read(open(filename, 'r'))
        self.vertices, self.faces = self._extract_plydata(plydata)


    def _extract_plydata(self, plydata):
        '''
        unpacks the structured array into standard np arrays
        ''' 
        vertices = plydata['vertex'].data
        np_vertex_data = vertices.view(np.float32).reshape(vertices.shape + (-1,))
        
        faces = np.zeros((plydata['face'].data.shape[0], 3), dtype=np.int32)
        for idx, t in enumerate(plydata['face'].data):
            faces[idx, :] = t[0]
            
        return np_vertex_data, faces

    def apply_transformation(self, trans):
        '''
        apply a 4x4 transformation matrix to the vertices
        '''
        n = self.vertices.shape[0]
        temp = np.concatenate((self.vertices, np.ones((n, 1))), axis=1).T
        temp_transformed = trans.dot(temp).T
        for idx in [0, 1, 2]:
            temp_transformed[:, idx] /= temp_transformed[:, 3]
        self.vertices = temp_transformed[:, :3]
        


class BigbirdMesh(Mesh):

    def load_bigbird(self, objname):
        '''
        loads a bigbird mesh
        '''
        mesh_path = paths.bigbird_folder + objname + "/meshes/poisson.ply"
        self.load_from_ply(mesh_path)


class Camera(object):
    '''
    this is a projective (depth?) camera class.
    Should be able to do projection
    '''

    def __init__(self):
        self.K = []
        self.H = []

    def set_intrinsics(self, K):
        self.K = K
        self.inv_K = np.linalg.inv(K)

    def set_extrinsics(self, H):
        '''extrinsics should be the location of the camera relative to the world origin'''
        self.H = H
        self.inv_H = np.linalg.inv(H)

    def load_extrinsics_from_dat(self, filepath):
        '''
        loads extrinsics from my own shitty dat file.
        in hindsight I should have written this as a yaml
        but it's too late now
        maybe
        '''
        f = open(filepath)
        f.readline() # throw away
        T = np.array(f.readline().split(' ')).astype(float)[np.newaxis, :].T
        f.readline() # throw away
        R = np.array([f.readline().strip().split(' '), 
                      f.readline().strip().split(' '), 
                      f.readline().strip().split(' ')]).astype(float)
        H = np.concatenate((R, T), axis=1)
        H = np.concatenate((H, np.array([0, 0, 0, 1])[np.newaxis, :]), axis=0)
        self.set_extrinsics(H)


    def adjust_intrinsic_scale(self, scale):
        '''
        changes the scaling, effectivly resixing the output image size
        '''
        self.K[0, 0] *= scale 
        self.K[1, 1] *= scale 
        self.K[0, 2] *= scale 
        self.K[1, 2] *= scale 


    def load_bigbird_matrices(self, modelname, imagename):
        '''
        loads the extrinsics and intrinsics for a bigbird camera
        camera name is something like 'NP5'
        '''
        cameraname, angle = imagename.split('_')

        # loading the pose and calibration files
        calib = h5py.File(paths.bigbird_folder + modelname + "/calibration.h5", 'r')
        pose_path = paths.bigbird_folder + modelname + "/poses/NP5_" + angle + "_pose.h5"
        pose = h5py.File(pose_path, 'r')

        # extracting extrinsic and intrinsic matrices
        np5_to_this_camera = np.array(calib['H_' + cameraname + '_from_NP5'])
        mesh_to_np5 = np.linalg.inv(np.array(pose['H_table_from_reference_camera']))

        intrinsics = np.array(calib[cameraname +'_rgb_K'])

        # applying to the camera
        self.set_extrinsics(np5_to_this_camera.dot(mesh_to_np5))
        self.set_intrinsics(intrinsics)


    def project_points(self, xyz):
        '''
        projects nx3 points xyz into the camera image
        returns their 2D projected location
        '''
        assert(xyz.shape[1] == 3)
        temp_xyz = self._apply_homo_transformation(xyz, self.H)
        raise Exception("unsure if H should be inverted - be careful")
        temp_trans = self._apply_transformation(temp_xyz, self.K)
        temp_trans[:, 0] /= temp_trans[:, 2]
        temp_trans[:, 1] /= temp_trans[:, 2]
        return temp_trans#[:, 0:2]


    def inv_project_points(self, uvd):
        '''
        throws u,v,d points in pixel coords (and depth)
        out into the real world, based on the transforms provided
        '''
        assert(uvd.shape[1] == 3)
        n_points = uvd.shape[0]

        # creating the camera rays
        uv1 = np.hstack((uvd[:, :2], np.ones((n_points, 1))))
        camera_rays = np.dot(self.inv_K, uv1.T).T

        # forming the xyz points in the camera coordinates
        temp = uvd[:, 2][np.newaxis, :].T
        xyz_at_cam_loc = temp * camera_rays

        # transforming points under the extrinsics
        xyz_in_world = self._apply_homo_transformation(xyz_at_cam_loc, self.H)
        xyz_in_world[:, 0] /= xyz_in_world[:, 3]
        xyz_in_world[:, 1] /= xyz_in_world[:, 3]
        xyz_in_world[:, 2] /= xyz_in_world[:, 3]
        return xyz_in_world[:, :3], xyz_at_cam_loc, uv1, camera_rays


    def _apply_transformation(self, xyz, trans):
        '''
        apply a 3x3 transformation matrix to the vertices
        '''
        to_add = np.zeros((3, 1))
        temp_trans = np.concatenate((trans, to_add), axis=1)
        return np.dot(temp_trans, xyz.T).T
        
    def _apply_homo_transformation(self, xyz, trans):
        '''
        apply a 4x4 transformation matrix to the vertices
        '''
        n = xyz.shape[0]
        temp = np.concatenate((xyz, np.ones((n, 1))), axis=1).T
        temp_transformed = trans.dot(temp).T
        return temp_transformed
        
    
