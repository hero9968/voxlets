'''
The idea here is to have a voxel class, which stores the prediction results.
Will happily construct the voxels from the front and back renders
In an ideal world, perhaps should inherit from a more generic voxel class
I haven't thought about this yet though...
'''

import cv2
import numpy as np
import paths
from scipy.ndimage.morphology import distance_transform_edt
import copy
from numbers import Number


class Voxels(object):
	'''
	voxel data base class - this will be parent of regular voxels and frustrum voxels
	'''
	def __init__(self, size, datatype):
		'''initialise the numpy voxel grid to the correct size'''
		assert np.prod(size) < 500e6 	# check to catch excess allocation
		self.V = np.zeros(size, datatype)


	def num_voxels(self):
		return np.prod(self.V.shape)


	def set_indicated_voxels(self, binary_array, values):
		'''
		helper function to set the values in V indicated in 
		the binary array to the values given in values.
		Question: Should values == length(binary_array) or 
		should values == sum(binary_array)??
		'''
		self.V[binary_array.reshape(self.V.shape)] = values


	def get_idxs(self, ijk):
		'''
		helper function to get the values indicated in the nx3 ijk array
		'''
		assert ijk.shape[1] == 3
		return self.V[ijk[:, 0], ijk[:, 1], ijk[:, 2]]


	def set_idxs(self, ijk, values):
		'''
		helper function to set the values indicated in the nx3 ijk array
		to the values in the n-long vector of values
		'''
		assert ijk.shape[1] == 3
		assert isinstance(values, Number) or ijk.shape[0] == values.shape[0]
		self.V[ijk[:, 0], ijk[:, 1], ijk[:, 2]] = values


	def find_valid_idx(self, idx):
		'''
		returns a logical array the same length as idx, with true
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
		helper function to extract the points referred 
		to by the 3D indices ;idxs
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
		#print np.min(trans_inside), np.max(trans_inside)
		trans_outside = distance_transform_edt(1-self.V.astype(float))
		#print np.min(trans_outside), np.max(trans_outside)
		#print np.min(trans_outside - trans_inside), np.max(trans_outside - trans_inside)

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




class WorldVoxels(Voxels):
	'''
	a regular grid of voxels in the real world.
	this now includes voxel size and the transformation from world
	space to the origin of the grid
	'''
	def __init__(self):
		pass


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


	def init_and_populate(self, indices):
		'''initialises the grid and populates, based on the indices in idx
		waits until now to initialise so it can be the correct size
		'''
		grid_size = np.max(indices, axis=0)+1
		print grid_size
		Voxels.__init__(self, grid_size, np.int8)
		#print indices[:, 0]
		self.V[indices[:, 0], indices[:, 1], indices[:, 2]] = 1


	def populate_from_vox_file(self, filepath):
		'''
		Loads 3d locations from my custom .vox file.
		My .vox file is almost but not quite a yaml
		Seemed that using pyyaml was incredibly slow so did this instead... bit of a hack!2
		'''
		f = open(filepath, 'r')
		f.readline() # origin:
		self.set_origin(np.array(f.readline().split(" ")).astype(float))
		f.readline() # extents:
		f.readline() # extents - don't care about this
		f.readline() # voxel_size:
		self.set_voxel_size(float(f.readline().strip()))
		f.readline() # vox:
		idx = np.array([line.split() for line in f]).astype(int)
		f.close()
		self.init_and_populate(idx)
		

	def idx_to_world(self, idx):
		'''
		converts an nx3 integer array, [i, j, k] coordinate to real-world 3D locations
		note that I do the 0.5 offset as I am treating idxs as voxel centres
		of course this should really be done in one homogeneoous transform...
		'''
		assert(idx.shape[1]==3)

		# applying scaling
		scaled_idx = (idx.astype(float)+0.5) * self.vox_size

		# rotate the grid points under the rotation R
#		if (count_grid.R == np.eye(3)).all():
#			scaled_rotated_idx = scaled_idx
#		else:	
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
		scaled_translated_xyz = translated_xyz / self.vox_size - 0.5

		# finally rotating 
		# note that (doing transpose twice seems to be quicker than np.dot(xyz, inv_R.T) )
#		if (self.inv_R == np.eye(3)).all():
#			scaled_translated_rotated_xyz = scaled_translated_xyz
#		else:
		scaled_translated_rotated_xyz = np.dot(self.inv_R, scaled_translated_xyz.T).T

		idx = (scaled_translated_rotated_xyz).astype(np.int)
	#	print self.origin
	#	print self.inv_R
	#	print self.vox_size

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
		'''
		assert(xyz.shape[1]==3)
		idxs, valid = self.world_to_idx(xyz, True)

		valid_idxs = idxs[valid, :]
		values = self.get_idxs(valid_idxs)

		return values, valid


	def idx_meshgrid(self):
		'''
		returns a meshgrid representation of the idx positions of every voxel in grid
		be careful if doing on large grids as can be memory expensive!
		'''
		has_cached = hasattr(self, '_cached_idx_meshgrid') and self._cached_idx_meshgrid.any()
		if not has_cached:
			# 0.5 offset beacuse we ant the centre of the voxels
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
		has_cached = hasattr(self, '_cached_idx_ij_meshgrid') and self._cached_idx_ij_meshgrid.any()
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
		has_cached = hasattr(self, '_cached_world_meshgrid') and self._cached_world_meshgrid.any()
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
		has_cached = hasattr(self, '_cached_world_xy_meshgrid') and self._cached_world_xy_meshgrid.any()
		if not has_cached:
			idx = self.idx_ij_meshgrid()
			self._cached_world_xy_meshgrid = self.idx_to_world(idx)

		return self._cached_world_xy_meshgrid



	def fill_from_grid(self, input_grid, method='naive', combine='replace'):
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
			self_world_xyz = self.world_meshgrid()
			self_idx = self.idx_meshgrid()

			# convert the indices to world xyz space
			#self_grid_in_sbox_idx, valid = input_grid.world_to_idx(self_world_xyz, True)
			#print "There are " + str(np.sum(valid)) + " valid voxels out of " + str(np.prod(valid.shape))
	
			#output_idxs = self_grid_in_sbox_idx[valid, :]
			#occupied_values = input_grid.extract_from_indices(output_idxs)

			# 2) Warp into idx space of input_grid and 
			# 3) See which are valid idxs in input_grid
			valid_values, valid = input_grid.just_valid_world_to_idx(self_world_xyz)
			#self.set_indicated_voxels(valid, occupied_values)
			
			# 4) Replace these values in self
			if combine == 'sum':
				addition = self.get_idxs(self_idx[valid, :])
				self.set_idxs(self_idx[valid, :], valid_values+addition)
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
			#print  world_xy_in_input_grid_idx[:, 0] >= 0
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

			# try saving up all the valid idxs
			#valid_ijs = np.empty((valid_k_idxs.shape[0]*valid_ij.shape[0], 3))
			#valid_data = []

			for world_slice_idx in valid_k_idxs:

				this_z_in_input_grid = world_z_in_input_grid_idx[world_slice_idx, 2]
				
				# don't bother extracting the data for this slice if we already have it cached in data_to_insert
				if current_shoebox_slice != this_z_in_input_grid:

					# extract the data from this slice in the input grid
					# yes we overwrite each time - but we don't care as we never use it again!
					valid_ij_in_input[:, 2] = this_z_in_input_grid

					data_to_insert = input_grid.get_idxs(valid_ij_in_input).astype(self.V.dtype)
					current_shoebox_slice = this_z_in_input_grid

				# now choosing where we put it in the world grid...
				# yes we overwrite each time - but we don't care as we never use it again!
				valid_ij[:, 2] = world_slice_idx

				# TODO - save all these up and do at end
				if combine=='accumulator':
					# here will do special stuff
					self.sumV[valid_ij[:, 0], valid_ij[:, 1], valid_ij[:, 2]] += data_to_insert
					self.countV[valid_ij[:, 0], valid_ij[:, 1], valid_ij[:, 2]] += 1

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
	



class BigBirdVoxels(WorldVoxels):

	def __init__(self):
		pass
		#self.WorldVoxels.__init__(self)

	def load_bigbird(self, modelname):
		idx_path = paths.base_path + "/bigbird_meshes/" + modelname + "/meshes/voxelised.vox"
		self.populate_from_vox_file(idx_path)



class VoxelGridCollection(object):
	'''
	class for doing things to a list of same-sized voxelgrids
	'''
	def __init__(self):
		pass


	def set_voxel_list(self, voxlist_in):
		self.voxlist = voxlist_in


	def cluster_voxlets(self, num_clusters, subsample_length):

		'''helper function to cluster voxlets'''

		# convert to np array
		all_sboxes = np.array([sbox.V.flatten() for sbox in self.voxlist]).astype(np.float16)

		# take subsample
		to_use_for_clustering = np.random.randint(0, all_sboxes.shape[0], size=(subsample_length))
		all_sboxes_subset = all_sboxes[to_use_for_clustering, :]
		print all_sboxes_subset.shape

		# doing clustering
		from sklearn.cluster import MiniBatchKMeans
		self.km = MiniBatchKMeans(n_clusters=num_clusters)
		self.km.fit(all_sboxes_subset)

		return self.km




class UprightAccumulator(WorldVoxels):
	'''
	accumulates multiple voxels into one output grid
	does the averaging etc also
	for this reason does it all in 32 bit floats
	Is an upright accumulator as assumed all the z directions are pointing the same way
	'''

	def __init__(self, gridsize):
		Voxels.__init__(self, gridsize, np.float32)
		self.grid_centre_from_grid_origin = []
		self.sumV = (copy.deepcopy(self.V)*0)
		self.countV = (copy.deepcopy(self.V)*0)


	def add_voxlet(self, voxlet):
		'''
		adds a single voxlet into the output grid
		'''

		# convert the indices to world xyz space
		#output_grid_in_voxlet_idx, valid = voxlet.world_to_idx(self.world_meshgrid(), True)

		#print "There are " + str(np.sum(valid)) + " valid voxels out of " + str(np.prod(valid.shape))

		# get the idxs in the output space and the values in the input space	    
		#output_idxs = output_grid_in_voxlet_idx[valid, :]
		#occupied_values = voxlet.extract_from_indices(output_idxs)
		
		#self.sumV[valid.reshape(self.V.shape)] += occupied_values
		#self.countV[valid.reshape(self.V.shape)] += 1
		self.fill_from_grid(voxlet, method='axis_aligned', combine='accumulator')
		

	def compute_average(self):
		'''
		computes a grid of the average values, stores in V
		'''
		self.countV[self.countV==0] = 100
		self.V = self.sumV / self.countV
		#self.V[np.isinf(self.V)] = np.nan
		return self.V



class ShoeBox(WorldVoxels):
	'''
	class for a 'shoebox' of voxels, which will ultimately surround a point and normal
	e.g. on a mesh or in a scene
	this is slightly complicated as it has a centre origin, as well as a grid location
	the rotation is the same for both
	'''

	def __init__(self, gridsize):
		Voxels.__init__(self, gridsize, np.int8)
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
		This is pretty uncharted territory now!
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
			raise Exception("R has det" % np.linalg.det(R))

		# computing the grid origin
		# using: p = R * p_from_grid_origin + grid_origin
		new_origin = point - np.dot(R, self.p_from_grid_origin)

		self.set_origin(new_origin, R)


class KinFuVoxels(Voxels):
	'''
	voxels as computed and save by KinFu
	'''
	def __init__(self):
		Voxels.__init__(self, (512, 512, 512), np.float32)


	def read_from_pcd(self, filename):
		''' 
		reads from an ascii pcd file
		'''
		fid = open(filename, 'r')

		# throw out header - doesn't have any interest
		for idx in range(11):
			fid.readline()

		# populate voxel grid
		for line in fid:
			t = line.split()
			self.V[int(t[0]), int(t[1]), int(t[2])] = float(t[3])


	def fill_full_grid(self):
		'''
		fills out the full voxel grid, given just a starting volume
		'''
		pass




'''
To think about - sparse voxel class?
'''


class FrustumGrid(Voxels):
	'''
	class for a frustum grid, such as that coming out of a camera
	For now this is hard-coded as uint8
	'''

	def __init__(self):
		pass


	def set_params(self, imagesize, d_front, d_back, focal_length):
		'''
		sets up the voxel grid. Also computes the number of voxels along the depth direction
		'''
		self.d_front = d_front
		self.d_back = d_back
		self.focal_length = focal_length
		self.imagesize = imagesize

		# work out the optimal number of voxels to put along the length
		self.m = np.ceil(2 * focal_length * (d_back - d_front) / (d_front + d_back))
		self.depth_vox_size = (d_back - d_front) / self.m

		gridsize = [imagesize[0], imagesize[1], self.m]
		Voxels.__init__(self, gridsize, np.float32)


	def vox_locations_in_camera_coords(self):
		'''
		returns a meshgrid representation of the camera coords of every
		voxel in the grid, in the coordinates of the camera
		'''
		#ax0 = np.arange(self.V.shape[0])
		#ax1 = np.arange(self.V.shape[1])
		#ax2 = np.arange(self.V.shape[2]) * self.m + self.d_front

		# 0.5 offset beacuse we ant the centre of the voxels
		A, B, C = np.mgrid[0.5:self.V.shape[0]+0.5, 
						   0.5:self.V.shape[1]+0.5,
						   0.5:self.V.shape[2]+0.5]

		C = C * self.depth_vox_size + self.d_front # scaling for depth
		grid = np.vstack((A.flatten(), B.flatten(), C.flatten()))
		return grid


	def depth_to_index(self, depth):
		''' 
		for a given real-world depth, computes the voxel index in the z direction, in {0,1,...,m}
		if the depth is out of range, returns -1
		'''
		#if depth > self.d_back or depth < self.d_front:
		#	return -1
		#else:
		scaled_depth = (depth - self.d_front) / (self.d_back - self.d_front)
		index = int(scaled_depth * self.m)
		#assert index >= 0 and index < self.V.shape[2]
		return index


	def populate(self, frontrender, back_predictions):
		'''
		fills the frustum grid with the predictions from the forest output
		could get this to redefine the d_back and d_front
		'''
		assert(frontrender.shape == self.imagesize)
		assert(back_predictions[0].shape == self.imagesize)

		# this is the addition to occupancy a vote from one tree makes
		per_tree_contribution = 1/float(len(back_predictions))

		# find the locations in frontrender which are non-nan
		non_nans = np.array(np.nonzero(~np.isnan(frontrender))).T

		# populate the voxel array
		for tree in back_predictions:
			for row, col in non_nans:
				#print row, col, frontrender[row, col], tree[row, col]
				start_idx = max(0, self.depth_to_index(frontrender[row, col]))
				end_idx = min(self.V.shape[2], self.depth_to_index(tree[row, col]))

				if np.isnan(start_idx) or np.isnan(end_idx):
					continue
				self.V[row, col, start_idx:end_idx] += per_tree_contribution


	def extract_warped_slice(self, slice_idx, output_image_height=500, gt=None):
		'''
		returns a horizontal slice warped to resemble the real-world positions
		maxdepth is the maximum distance to consider in the real-world coordinates
		'''
		# computing the perspective transform between voxel space and output image space
		h, w, = self.imagesize
		pts1 = np.float32([[0,0],[w,0],[0,h],[w,h]])

		# change in scale between real-world and this output image
		scale = output_image_height / (self.d_back - self.d_front)
		output_image_width = scale * (w * self.d_back) / self.focal_length
		frustum_front_width = scale * (w * self.d_front) / self.focal_length

		pts2 = np.float32([[0,0], 
						   [output_image_width,0], 
						   [output_image_width/2 - frustum_front_width/2, output_image_height], 
						   [output_image_width/2 + frustum_front_width/2, output_image_height]])

		M = cv2.getPerspectiveTransform(pts1,pts2)

		# extracting and warping slice
		output_size = (int(output_image_width), int(output_image_height))

		return cv2.warpPerspective(self.V[:, :, slice_idx], M, output_size, borderValue=1)




class SliceFiller(object):
	'''
	fills a slice of voxels 
	not really sure what I'm doing here but something might work out..
	'''

	def __init__(self, front_slice, thickness_predictions, focal_length):
		self.front_slice = front_slice
		self.thickness_predictions = thickness_predictions
		self.focal_length = focal_length


	def fill_slice(self, start_depth, end_depth, gt=None):
		'''
		similar to fill_voxels but only doing one slice at a time
		'''

		# finding the start and end depths...
		#start_depth = self.get_start_depth()
		#end_depth = self.get_end_depth()
		scale_factor = (start_depth+end_depth) / (30 * self.focal_length)

		# create the volume...
		depth_in_voxels = int(np.ceil((end_depth - start_depth) / scale_factor))
		thisslice = np.zeros((depth_in_voxels, self.front_slice.shape[0]))
		print "This slice shape is " + str(thisslice.shape)
		print "Depth in vocels is " + str(depth_in_voxels)
		print "Start depth is " + str(start_depth)
		number_trees = self.thickness_predictions.shape[0]

		#max_depth_so_far = 0
		#min_depth_so_far = 10000

		# to do 
		# 1) use indices of the mask 
		# 2) convert front and back render to voxel coords before loop
		# 3) check for inside and outside the volume before the loop

		# populate the volume with each depth prediction...
		row = self.front_slice
		for colidx, front_val in enumerate(row):

			front_vox_depth = round((front_val - start_depth) / scale_factor)

			for tree_idx in range(number_trees):

				thickness = self.thickness_predictions[tree_idx, 0, colidx]

				if ~np.isnan(front_val) and ~np.isnan(thickness):

					back_vox_depth = round((thickness) / scale_factor)
					thisslice[int(front_vox_depth):int(front_vox_depth+back_vox_depth), colidx] += 1
			
			# adding gt
			if False:# any(gt):
				thickness = gt[colidx]

				if ~np.isnan(front_val) and ~np.isnan(thickness):
					back_vox_depth = round((thickness - start_depth) / scale_factor)
					thisslice[int(front_vox_depth+back_vox_depth), colidx] = 1.5*number_trees
					thisslice[int(front_vox_depth+back_vox_depth+1), colidx] = 1.5*number_trees
					thisslice[int(front_vox_depth+back_vox_depth-1), colidx] = 1.5*number_trees


		#self.thisslice = vol
		return thisslice


	def extract_warped_slice(self, mindepth, maxdepth, output_image_height=500, gt=None):
		'''
		returns a horizontal slice warped to resemble the real-world positions
		maxdepth is the maximum distance to consider in the real-world coordinates
		'''
		#mindepth = self.get_start_depth()
		#if maxdepth == -1:
	#		maxdepth = self.get_end_depth()
		scale_factor = output_image_height / (maxdepth - mindepth)
		volume_slice = self.fill_slice(mindepth, maxdepth, gt)
		self.volume_slice = volume_slice

		# computing the perspective transform between voxel space and output image space
		h = volume_slice.shape[0]
		w = volume_slice.shape[1]
		pts1 = np.float32([[0,0],[w,0],[0,h],[w,h]])

		d1 = scale_factor * (mindepth * float(volume_slice.shape[0])/2) / self.focal_length
		d2 = scale_factor * (maxdepth * float(volume_slice.shape[1])/2) / self.focal_length
		pts2 = np.float32([[d2-d1,0], [d1+d2,0], [0,output_image_height], [d2*2,output_image_height]])

		M = cv2.getPerspectiveTransform(pts1,pts2)

		# extracting and warping slice
		output_size = (2*int(d2),int(output_image_height))

		warped_image = cv2.warpPerspective(volume_slice,M,output_size, borderValue=1)
		return warped_image



