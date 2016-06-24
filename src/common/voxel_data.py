# Standard and plotting
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import sys
import numpy as np
import copy
from numbers import Number
import subprocess as sp

# IO
import cPickle as pickle
import yaml
from scipy.io import loadmat, savemat

# Image and voxel manipulation
from scipy.ndimage.morphology import distance_transform_edt

# Custom
import mesh


def load_voxels(loadpath):
    with open(loadpath, 'rb') as f:
        return pickle.load(f)


class Voxels(object):
    '''
    voxel data base class - this will be parent of regular voxels and frustrum voxels
    '''

    def __init__(self, size, datatype):
        '''Initialise the numpy voxel grid to the correct size'''
        assert np.prod(size) < 500e6    # check to catch excess allocation
        self.V = np.zeros(size, datatype)

    @classmethod
    def load_from_mat(cls, mat_path):
        '''
        Initialises and loads from a mat file
        '''
        vox = cls()
        vox.__dict__ = loadmat(mat_path)
        return vox

    def save_to_mat(self, mat_path, clear_cache=True):
        '''
        Dumps internals to a mat file
        '''
        if clear_cache:
            self._clear_cache()

        savemat(mat_path, self.__dict__)

    def copy(self):
        '''Returns a deep copy of self'''
        return copy.deepcopy(self)

    def blank_copy(self):
        '''Returns a copy of self, but with all voxels empty'''
        temp = copy.deepcopy(self)
        # not using temp.V*=0 in case nans are present
        temp.V = np.zeros(temp.V.shape, temp.V.dtype)
        return temp

    def num_voxels(self):
        return np.prod(self.V.shape)

    def set_indicated_voxels(self, binary_array, values):
        '''
        Helper function to set the values in V indicated in
        the binary array to the values given in values.
        Question: Should values == length(binary_array) or
        should values == sum(binary_array)??
        '''
        self.V[binary_array.reshape(self.V.shape)] = values

    def get_indicated_voxels(self, binary_array):
        '''
        Helper function to set the values in V indicated in
        the binary array to the values given in values.
        Question: Should values == length(binary_array) or
        should values == sum(binary_array)??
        '''
        return self.V[binary_array.reshape(self.V.shape)]

    def get_idxs(self, ijk, check_bounds=False):
        '''
        Helper function to get the values indicated in the nx3 ijk array.
        If check_bounds, then returns a nan for each out of range location
        '''
        assert ijk.shape[1] == 3
        if check_bounds:
            valid = self.find_valid_idx(ijk)
            output = np.zeros(ijk.shape[0]) * np.nan
            output[valid] = self.V[ijk[valid, 0], ijk[valid, 1], ijk[valid, 2]]
            return output
        else:
            return self.V[ijk[:, 0], ijk[:, 1], ijk[:, 2]]

    def set_idxs(self, ijk, values, check_bounds=False):
        '''
        Helper function to set the values indicated in the nx3 ijk array
        to the values in the n-long vector of values
        '''
        assert ijk.shape[1] == 3
        assert isinstance(values, Number) or ijk.shape[0] == values.shape[0]

        if check_bounds:
            valid = self.find_valid_idx(ijk)
            if isinstance(values, Number):
                self.V[ijk[valid, 0], ijk[valid, 1], ijk[valid, 2]] = values
            else:
                self.V[ijk[valid, 0], ijk[valid, 1], ijk[valid, 2]] = values[valid]
        else:
            self.V[ijk[:, 0], ijk[:, 1], ijk[:, 2]] = values

    def find_valid_idx(self, idx):
        '''
        Returns a logical array the same length as idx, with true
        where idx is within the range of self.V, and false otherwise
        '''
        return np.logical_and.reduce((idx[:, 0] < self.V.shape[0],
                                       idx[:, 0] >= 0,
                                       idx[:, 1] < self.V.shape[1],
                                       idx[:, 1] >= 0,
                                       idx[:, 2] < self.V.shape[2],
                                       idx[:, 2] >= 0))

    def extract_from_indices(self, idxs, check_bounds=False):
        '''
        Helper function to extract the points referred
        to by the 3D indices 'idxs'
        If check_bound is true, then will check all the idxs to see if they are valid
        invalid idxs will get a nan returned
        (Could also make it so invalid idxs don't get anything returned...)
        '''
        assert idxs.shape[1] == 3

        if check_bounds:
            print "Warning - not tested this bit yet"
            # create output array, find which are valid idxs and look up their values
            output_array = np.nan(idxs.shape[0], 3)
            valid_idxs = self.find_valid_idx(idx)
            output_array[valid_idxs] = self.V[idxs[valid_idxs, 0],
                                              idxs[valid_idxs, 1],
                                              idxs[valid_idxs, 2]]
            return output_array
        else:
            return self.V[idxs[:, 0], idxs[:, 1], idxs[:, 2]]

    def get_valid_values(self, idxs):
        '''
        Sees which idxs are in range of the voxel grid.
        Returns values from V for all the indices which were in range
        Also returns a binary array indicating which idx it managed to extract from
        '''
        pass

    def get_corners(self):
        '''
        returns 8x3 array of the corners of the voxel grid in world space coordinates
        '''
        corners = []
        for i in [0, self.V.shape[0]]:
            for j in [0, self.V.shape[1]]:
                for k in [0, self.V.shape[2]]:
                    corners.append([i, j, k])

        return self.idx_to_world(np.array(corners))

    def compute_sdt(self):
        '''
        returns a signed distance transform the same size as V
        uses scipy's Euclidean distance transform
        '''
        trans_inside = distance_transform_edt(self.V.astype(float))
        trans_outside = distance_transform_edt(1-self.V.astype(float))
        return trans_outside - trans_inside

    def compute_tsdf(self, truncation):
        '''
        computes tsdf in real world units
        truncation limit (mu in kinfupaper) needs to be set
        '''
        sdf = self.compute_sdt()

        # convert to real-world distances
        sdf *= self.vox_size

        # truncate
        sdf[sdf > truncation] = truncation
        sdf[sdf < -truncation] = -truncation

        return sdf

    def convert_to_tsdf(self, truncation):
        '''
        converts binary grid to a tsdf
        '''
        self.V = self.compute_tsdf(truncation).astype(np.float16)

    def save(self, filename):
        '''
        Serialisation routine. Using the highest protocol of pickle makes this
        quick and efficient. However, must clear the cache first!
        '''
        self._clear_cache()

        with open(filename, 'wb') as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)


class WorldVoxels(Voxels):
    '''
    a regular grid of voxels in the real world.
    this now includes voxel size and the transformation from world
    space to the origin of the grid
    '''
    def __init__(self):
        pass

    @classmethod
    def load_from_dat(cls, dat_file, meta_yaml_file):
        '''
        Initialise and load data from a binary dat file, and load
        metadata from a yaml file. This is a very efficient file format
        when compared to pkl, mat etc
        '''
        meta = yaml.load(open(meta_yaml_file))
        vox = cls()
        vox.R = np.array(meta['R']).reshape((3, 3))
        vox.origin = np.array(meta['T'])
        vox.vox_size = meta['voxelsize']
        vox.V = np.fromfile(dat_file, dtype=np.float16).reshape(meta['shape'])
        return vox

    def init_and_populate(self, indices):
        '''
        Initialises the grid and populates, based on the indices in idx
        Waits until now to initialise so it can be the correct size
        '''
        grid_size = np.max(indices, axis=0)+1
        Voxels.__init__(self, grid_size, np.int8)
        self.V[indices[:, 0], indices[:, 1], indices[:, 2]] = 1

    def set_voxel_size(self, vox_size):
        '''should be scalar'''
        self.vox_size = vox_size
        self._clear_cache()

    def set_origin(self, origin, rotation=np.eye(3)):
        '''
        setting the origin in world space
        the origin is a 3-long vector denoting the position of the grid
        the voxel grid is transformed from the origin by a rotation followed by this translation
        the rotation is a 3x3 matrix R
        '''
        origin = np.array(origin)
        assert origin.shape[0] == 3
        self.origin = origin
        assert rotation.shape[0]==3 and rotation.shape[1]==3
        self.R = rotation
        self.inv_R = np.linalg.inv(rotation)

        self._clear_cache()

    def _clear_cache(self):
        '''
        clears cached items, should call this after a change in origin or rotation etc
        '''
        if hasattr(self, '_cached_world_meshgrid'):
            self._cached_world_meshgrid = []

        if hasattr(self, '_cached_idx_meshgrid'):
            self._cached_idx_meshgrid = []

    def idx_to_world(self, idx):
        '''
        converts an nx3 integer array, [i, j, k] coordinate to real-world 3D locations
        note that I do the 0.5 offset as I am treating idxs as voxel centres
        of course this should really be done in one homogeneoous transform...
        '''
        assert(idx.shape[1]==3)

        # applying scaling
        scaled_idx = (idx.astype(float)+0.5) * self.vox_size
        scaled_rotated_idx = np.dot(self.R, scaled_idx.T).T

        # applying the real-world offset
        world_coords = scaled_rotated_idx + self.origin

        return world_coords

    def world_to_idx(self, xyz, detect_out_of_range=False):
        '''
        converts an nx3 world coordinates, [x, y, z], to ijk locations
        if detect_out_of_range is true, then also returns a logical array saying which are
        in range of the grid locations
        '''
        assert(xyz.shape[1]==3)

        # applying translate
        translated_xyz = xyz - self.origin

        # ...scaling
        scaled_translated_xyz = translated_xyz / self.vox_size

        # finally rotating
        # note that (doing transpose twice seems to be quicker than np.dot(xyz, inv_R.T) )
        scaled_translated_rotated_xyz = np.dot(self.inv_R, scaled_translated_xyz.T).T

        idx = np.floor(scaled_translated_rotated_xyz).astype(np.int)

        if detect_out_of_range:
            valid = self.find_valid_idx(idx)
            return idx, valid
        else:
            return idx

    def idx_to_world_transform4x4(self):
        '''
        returns a 4x4 transform from idx to world locations based on the various things
        assumes rotation followed by a translation
        '''
        scale_factor = self.vox_size
        translation = self.origin[np.newaxis, :]

        half = np.concatenate((scale_factor * self.R, translation.T), axis=1)
        full = np.concatenate((half, np.array([[0, 0, 0, 1]])), axis=0)
        return full

    def just_valid_world_to_idx(self, xyz, detect_out_of_range=False):
        '''
        as world_to_idx, but only returns the values of the valid idx locations,
        also returns a binary array indicating which these were
        valid_idxs are the locations of the values from self which were pulled out
        '''
        assert(xyz.shape[1]==3)
        idxs, valid = self.world_to_idx(xyz, True)

        valid_idxs = idxs[valid, :]
        values = self.get_idxs(valid_idxs)

        return values, valid, valid_idxs

    def idx_meshgrid(self):
        '''
        returns a meshgrid representation of the idx positions of every voxel in grid
        be careful if doing on large grids as can be memory expensive!
        '''
        has_cached = hasattr(self, '_cached_idx_meshgrid') and \
            len(self._cached_idx_meshgrid) > 0 and \
            self._cached_idx_meshgrid.any()
        if not has_cached:
            # 0.5 offset beacuse we want the centre of the voxels
            A, B, C = np.mgrid[0:self.V.shape[0],
                               0:self.V.shape[1],
                               0:self.V.shape[2]]

            #C = C * self.depth_vox_size + self.d_front # scaling for depth
            self._cached_idx_meshgrid = \
                np.vstack((A.flatten(), B.flatten(), C.flatten())).T

        return self._cached_idx_meshgrid

    def idx_ij_meshgrid(self):
        '''
        returns a meshgrid representation of the idx positions of every voxel in grid
        be careful if doing on large grids as can be memory expensive!
        '''
        has_cached = hasattr(self, '_cached_idx_ij_meshgrid') and \
            len(self._cached_idx_ij_meshgrid) > 0 and \
            self._cached_idx_ij_meshgrid.any()
        if not has_cached:
            # 0.5 offset beacuse we ant the centre of the voxels
            A, B = np.mgrid[0:self.V.shape[0], 0:self.V.shape[1]]

            self._cached_idx_ij_meshgrid = \
                np.vstack((A.flatten(), B.flatten(), (0*A).flatten())).T

        return self._cached_idx_ij_meshgrid

    def world_meshgrid(self):
        '''
        returns meshgrid representation of all the xyz positions of every point
        in the grid, transformed into world space!
        Makes use of caching which could be dangerous... be careful!
        '''
        has_cached = hasattr(self, '_cached_world_meshgrid') and \
            len(self._cached_world_meshgrid) > 0 and \
            self._cached_world_meshgrid.any()
        if not has_cached:
            idx = self.idx_meshgrid()
            self._cached_world_meshgrid = self.idx_to_world(idx)

        return self._cached_world_meshgrid

    def world_xy_meshgrid(self):
        '''
        returns meshgrid representation of all the xyz positions of every point
        in the grid, transformed into world space!
        Makes use of caching which could be dangerous... be careful!
        '''
        has_cached = hasattr(self, '_cached_world_xy_meshgrid') and \
            len(self._cached_world_xy_meshgrid) > 0 and \
            self._cached_world_xy_meshgrid.any()
        if not has_cached:
            idx = self.idx_ij_meshgrid()
            self._cached_world_xy_meshgrid = self.idx_to_world(idx)

        return self._cached_world_xy_meshgrid

    def fill_from_grid(self, input_grid, method='naive', combine='replace',
            weights=None, keep_explicit_count=False):
        '''
        warps input_grid into the world space of self.
        For all voxels in self.V which input_grid overlaps with,
        replace the voxel value with the corresponding value in input_grid.V
        'combine' can be sum (add onto existing elements) or replace (overwirte existing elements)
        '''

        if method=='naive':
            '''
            This is slow as need to transform twice,
            and transform *all* the voxels in self
            '''
            # 1) Compute world meshgrid for self
            self_idx = self.idx_meshgrid()
            self_world_xyz = self.world_meshgrid()

            # 2) Warp into idx space of input_grid and
            # 3) See which are valid idxs in input_grid
            valid_values, valid, _ = input_grid.just_valid_world_to_idx(self_world_xyz)

            # 4) Replace these values in self
            if combine == 'sum':
                addition = self.get_idxs(self_idx[valid, :])
                self.set_idxs(self_idx[valid, :], valid_values+addition)
            elif combine=='accumulator':
                self.sumV[valid.reshape(self.sumV.shape)] += valid_values
                self.countV[valid.reshape(self.countV.shape)] += 1
            else:
                self.set_idxs(self_idx[valid, :], valid_values)

        elif method=='bounding_box':
            '''
            First transform the bounding box of input grid into self to see which
            voxels from self need to transform into the space of input grid
            '''
            raise Exception("Not implemented yet")

        elif method=='axis_aligned':
            '''
            Assumes transformation is aligned with an axis -
            perhaps always the z-axis.
            Should end up being efficient as need only to do the complex
            transformation for one slice, then can re-use this for all the others
            '''
            world_ij = self.idx_ij_meshgrid() # instead, just get an ij slice and convert here...

            world_xy = self.idx_to_world(world_ij)

            world_xy_in_input_grid_idx = input_grid.world_to_idx(world_xy)

            #print input_grid.V.shape

            # now see which of these are valid...
            valid_ij_logical = np.logical_and.reduce((world_xy_in_input_grid_idx[:, 0] >= 0,
                                          world_xy_in_input_grid_idx[:, 0] < input_grid.V.shape[0],
                                          world_xy_in_input_grid_idx[:, 1] >= 0,
                                          world_xy_in_input_grid_idx[:, 1] < input_grid.V.shape[1]))

            valid_ij = world_ij[valid_ij_logical, :]
            valid_ij_in_input = world_xy_in_input_grid_idx[valid_ij_logical, :]

            # now find the valid slices in self: need to know the mapping along the z-axis
            world_k_col, world_z_col = self.get_z_locations()

            # each height in the input grid locations...
            world_z_in_input_grid_idx = input_grid.world_to_idx(world_z_col)

            # valid denotes which rows in world are happy to be filled by the input grid
            valid = np.logical_and(world_z_in_input_grid_idx[:, 2] >= 0,
                                   world_z_in_input_grid_idx[:, 2] < input_grid.V.shape[2])

            # now fill in slice by slice...
            valid_k_idxs = np.array(np.nonzero(valid)[0])

            # keeps track of what the current slice in the shoebox is
            current_shoebox_slice = np.nan

            for world_slice_idx in valid_k_idxs:

                this_z_in_input_grid = world_z_in_input_grid_idx[world_slice_idx, 2]

                # don't bother extracting the data for this slice if we already have it cached in data_to_insert
                if current_shoebox_slice != this_z_in_input_grid:

                    # extract the data from this slice in the input grid
                    # yes we overwrite each time - but we don't care as we never use it again!
                    valid_ij_in_input[:, 2] = this_z_in_input_grid

                    data_to_insert = input_grid.get_idxs(valid_ij_in_input).astype(self.V.dtype)
                    current_shoebox_slice = this_z_in_input_grid

                    if weights:
                        weights_to_insert = weights.get_idxs(valid_ij_in_input).astype(self.V.dtype)

                # now choosing where we put it in the world grid...
                # yes we overwrite each time - but we don't care as we never use it again!
                valid_ij[:, 2] = world_slice_idx

                # TODO - save all these up and do at end
                if combine=='accumulator':
                    # here will do special stuff
                    if weights:
                        self.sumV[valid_ij[:, 0], valid_ij[:, 1], valid_ij[:, 2]] += data_to_insert * weights_to_insert
                        self.countV[valid_ij[:, 0], valid_ij[:, 1], valid_ij[:, 2]] += weights_to_insert
                    else:
                        self.sumV[valid_ij[:, 0], valid_ij[:, 1], valid_ij[:, 2]] += data_to_insert
                        self.countV[valid_ij[:, 0], valid_ij[:, 1], valid_ij[:, 2]] += 1

                    if keep_explicit_count:
                        self.explicit_countV[valid_ij[:, 0], valid_ij[:, 1], valid_ij[:, 2]] += 1

                elif combine == 'sum':
                    addition = self.get_idxs(valid_ij)
                    self.set_idxs(valid_ij, data_to_insert+addition)

                elif combine=='replace':
                    self.set_idxs(valid_ij, data_to_insert)

                else:
                    raise Exception("unknown combine type")

            valid_ij = None # to prevent acciental use again
            valid_ij_in_input = None # to prevent acciental use again

        else:
            raise Exception("Unknown transformation method")

    def get_z_locations(self):
        '''
        returns a matrix of all the z locations in this image
        '''
        world_k = np.arange(0, self.V.shape[2])
        extra_zeros = np.zeros((world_k.shape[0], 2))
        world_k_col = np.concatenate((extra_zeros, world_k[:, np.newaxis]), axis=1)

        world_z_col = self.idx_to_world(world_k_col)

        return world_k_col, world_z_col

    def render_view(self, savepath, xy_centre=None, ground_height=None,
            keep_obj=False, actually_render=True, flip=False):
        '''
        render a single view of a voxel grid, using blender...
        ground height is in meters
        '''
        temp = self.copy()

        # put in a ground plane...
        if ground_height:
            height_voxels = float(ground_height) / float(temp.vox_size)
            temp.V[:, :, :height_voxels] = -10
            temp_slice = temp.V[:, :, height_voxels]
            temp_slice[np.isnan(temp_slice)] = 10
            temp.V[:, :, height_voxels] = temp_slice

        #pickle.dump(self, open('/tmp/temp_voxel_grid.pkl', 'w'), protocol=pickle.HIGHEST_PROTOCOL)
        print "Generating mesh...",
        sys.stdout.flush()
        ms = mesh.Mesh()
        ms.from_volume(temp, 0)
        #pickle.dump(ms, open('/tmp/temp_mesh.pkl', 'w'), protocol=pickle.HIGHEST_PROTOCOL)
        ms.remove_nan_vertices()

        if xy_centre:
            cen = temp.origin + (np.array(temp.V.shape) * temp.vox_size) / 2.0
            ms.vertices[:, :2] -= cen[:2]
            ms.vertices[:, 2] -= 0.05

        if flip:
            ms.vertices[:, 0] *= -1

        print "Writing to obj...",
        sys.stdout.flush()
        ms.write_to_obj(savepath + '.obj')

        print "Rendering...",
        if actually_render:
            sys.stdout.flush()
            blend_path = os.path.expanduser('~/projects/shape_sharing/src/rendered_scenes/spinaround/spin.blend')
            blend_py_path = os.path.expanduser('~/projects/shape_sharing/src/rendered_scenes/spinaround/blender_spinaround_frame.py')
            subenv = os.environ.copy()
            subenv['BLENDERSAVEFILE'] = savepath
            sp.call(['blender',
                     blend_path,
                     "-b", "-P",
                     blend_py_path],
                     env=subenv,
                     stdout=open(os.devnull, 'w'),
                     close_fds=True)

        print "Done rendering...",
        sys.stdout.flush()

        if not keep_obj:
            os.remove(savepath + '.obj')

    def project_unobserved_voxels(self, im):
        # project the nan voxels from grid into the image...
        to_project_idxs = np.where(self.V.flatten() != np.nanmax(self.V))[0]
        nan_xyz = self.world_meshgrid()[to_project_idxs]
        return im.cam.project_points(nan_xyz).astype(np.int32), to_project_idxs

    def plot_slices(self, savepath):
        '''
        plots the slices through the grid to subplots
        '''
        height = self.V.shape[2]
        mu = max(np.abs(np.nanmin(self.V)), np.nanmax(self.V))

        # don't bother getting slices at the extremes of the volume...
        slice_locations = np.linspace(0, height, 9+2)[1:-1]
        for count, idx in enumerate(slice_locations):
            plt.subplot(3, 3, count+1)
            plt.imshow(self.V[:, :, idx])
            plt.clim(-mu, mu)
            plt.title("idx = %d" % idx)

        plt.savefig(savepath)


class UprightAccumulator(WorldVoxels):
    '''
    accumulates multiple voxels into one output grid
    does the averaging etc also
    for this reason does it all in 32 bit floats
    Is an upright accumulator as assumed all the z directions are pointing the same way
    '''

    def __init__(self, gridsize, keep_explicit_count=False):
        '''
        keep_explicit_count is an option, if true we keep an integer count
        of how many predictions have been made at each voxel
        useful if countV ends up just being a sum of weights, in which case
        it is pretty hard to find out how many predictions are made everywhere
        '''
        Voxels.__init__(self, gridsize, np.float32)
        self.grid_centre_from_grid_origin = []
        self.sumV = (copy.deepcopy(self.V)*0)
        self.countV = (copy.deepcopy(self.V)*0)

        if keep_explicit_count:
            self.explicit_countV = (copy.deepcopy(self.V)*0)

        self.keep_explicit_count = keep_explicit_count

    def add_voxlet(self, voxlet, accum_only_predict_true, weights=None):
        '''
        adds a single voxlet into the output grid

        accum_only_predict_true
            if true, then only add in the 'occupied' and the narrow band from the
            prediction.
        '''
        if accum_only_predict_true:

            self_world_xyz = self.world_meshgrid()
            self_idx = self.idx_meshgrid()

            # 2) Warp into idx space of input_grid and
            # 3) See which are valid idxs in input_grid
            data_to_insert, valid, _ = voxlet.just_valid_world_to_idx(self_world_xyz)

            # now only use the values which pass the test...
            valid_data = data_to_insert < np.nanmax(voxlet.V)
            data_to_insert = data_to_insert[valid_data]
            valid = np.where(valid)[0][valid_data]

            # 4) Replace these values in self
            # do this manually...
            self.sumV[self_idx[valid, 0], self_idx[valid, 1], self_idx[valid, 2]] += data_to_insert
            self.countV[self_idx[valid, 0], self_idx[valid, 1], self_idx[valid, 2]] += 1

        elif weights is not None:
            # # Doing the accumulation in a naive way here...
            weights_grid = voxlet.blank_copy()
            weights_grid.V = weights.reshape(weights_grid.V.shape)
            self.fill_from_grid(voxlet, method='axis_aligned',
                combine='accumulator', weights=weights_grid,
                keep_explicit_count=self.keep_explicit_count)

        else:
            self.fill_from_grid(voxlet, method='axis_aligned', combine='accumulator')

    def compute_average(self, nan_value=0):
        '''
        computes a grid of the average values, stores in V
        '''
        nan_locations = self.countV==0
        temp_countV = copy.deepcopy(self.countV)
        temp_countV[nan_locations] = 100  # to avoid div by zero error
        self.V = self.sumV / temp_countV
        self.V[nan_locations] = np.nan #nan_value
        #self.V[np.isinf(self.V)] = np.nan

        # clear these grid for memory reasons
        #self.sumV = None
        #self.countV = None

        # return myself
        return self


class ShoeBox(WorldVoxels):
    '''
    class for a 'shoebox' of voxels, which will ultimately surround a point and normal
    e.g. on a mesh or in a scene
    this is slightly complicated as it has a centre origin, as well as a grid location
    the rotation is the same for both
    '''

    def __init__(self, gridsize, data_type=np.int8):
        Voxels.__init__(self, gridsize, data_type)
        self.grid_centre_from_grid_origin = []

    def set_p_from_grid_origin(self, p_from_grid_origin):
        '''
        the 'p' is an arbitrary position -
        it does not actually have to be in the centre of the grid!
        p_from_grid_origin is computed on the unrotated grid,
        but in units of world space -
        so could be e.g. [0.2m, 0.2, 0.2m]
        '''
        assert p_from_grid_origin.shape[0] == 3
        self.p_from_grid_origin = p_from_grid_origin

    def initialise_from_point_and_normal(self, point, normal, updir):
        '''
        assumes already set 'voxelsize', also 'grid_centre_from_grid_origin',
        also the actual size of the grid
        '''
        assert updir.shape[0] == 3
        assert normal.shape[0] == 3
        assert point.shape[0] == 3

        # creating the rotation matrix
        new_z = updir
        new_x = np.cross(updir, normal)
        new_x /= np.linalg.norm(new_x)
        new_y = np.cross(updir, new_x)
        new_y /= np.linalg.norm(new_y)
        R = np.vstack((new_x, new_y, new_z)).T

        assert R.shape[0] == 3 and R.shape[1] == 3

        if np.abs(np.linalg.det(R) - 1) > 0.00001:
            print "R is " + str(R)
            print updir, normal, point
            raise Exception("R has det" % np.linalg.det(R))

        # computing the grid origin
        # using: p = R * p_from_grid_origin + grid_origin
        new_origin = point - np.dot(R, self.p_from_grid_origin)

        self.set_origin(new_origin, R)
