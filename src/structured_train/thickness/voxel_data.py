'''
The idea here is to have a voxel class, which stores the prediction results.
Will happily construct the voxels from the front and back renders
In an ideal world, perhaps should inherit from a more generic voxel class
I haven't thought about this yet though...
'''

import cv2
import numpy as np


class Voxels(object):
	'''
	voxel data base class - this will be parent of regular voxels and frustrum voxels
	'''
	def __init__(self, size, datatype):
		'''initialise the numpy voxel grid to the correct size'''
		assert np.prod(size) < 500e6 	# check to catch excess allocation
		self.V = np.zeros(size, datatype)


	def sum_positive(self):
		return np.sum(self.V > 0)


	def num_voxels(self):
		return np.prod(self.V.shape)


class WorldVoxels(Voxels):
	'''
	a regulare grid of voxels in the real world
	this now includes voxel size
	and the translation to the origin of the grid
	'''
	def __init__(self):
		pass


	def set_voxel_size(self, vox_size):
		'''should be scalar'''
		self.vox_size = vox_size


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


	def init_and_populate(self, indices):
		'''initialises the grid and populates, based on the indices in idx
		waits until now to initialise so it can be the correct size
		'''
		grid_size = np.max(indices, axis=0)+1
		print grid_size
		Voxels.__init__(self, grid_size, np.int8)
		print indices[:, 0]
		self.V[indices[:, 0], indices[:, 1], indices[:, 2]] = 1


	def populate_from_txt_file(self, filepath):
		'''
		loads 3d locations from txt file
		'''
		pass


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

		# applying translate (SHOULD ALSO DO ROTATION...)
		translated_xyz = xyz - self.origin

		# ...scaling
		scaled_translated_xyz = translated_xyz / self.vox_size - 0.5

		# finally rotating
		scaled_translated_rotated_xyz = np.dot(self.inv_R, scaled_translated_xyz.T).T
		idx = np.round(scaled_translated_rotated_xyz).astype(np.int)
		
		if detect_out_of_range:
			valid = np.logical_and.reduce((idx[:, 0] < self.V.shape[0],
										   idx[:, 0] >= 0,
										   idx[:, 1] < self.V.shape[1],
										   idx[:, 1] >= 0,
										   idx[:, 2] < self.V.shape[2],
										   idx[:, 2] >= 0))
			return idx, valid
		else:
			return idx


	def idx_meshgrid(self):
		'''
		returns a meshgrid representation of the idx positions of every voxel in grid
		be careful if doing on large grids as can be memory expensive!
		'''
		# 0.5 offset beacuse we ant the centre of the voxels
		A, B, C = np.mgrid[0:self.V.shape[0], 
						   0:self.V.shape[1],
						   0:self.V.shape[2]]

		C = C * self.depth_vox_size + self.d_front # scaling for depth
		grid = np.vstack((A.flatten(), B.flatten(), C.flatten()))
		return grid


	def world_meshgrid(self):
		'''
		returns meshgrid representation of all the xyz positions of every point
		in the grid, transformed into world space!
		'''
		idx = self.idx_meshgrid()
		return self.idx_to_world(idx)



class ShoeBox(WorldVoxels):
	'''
	class for a 'shoebox' of voxels, which will ultimately surround a point and normal
	e.g. on a mesh or in a scene
	this is slightly complicated as it has a centre origin, as well as a grid location
	the rotation is the same for both
	'''

	def __init__(self, gridsize):
		self.WorldVoxels.Voxels.__init__(gridsize, np.int)
		self.grid_centre_from_grid_origin = []


	def set_p_from_grid_origin(self, p_from_grid_origin):
		'''
		the 'p' is an arbitray position - 
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

		# creating the rotation matrix
		new_z = updir
		new_x = np.cross(normal, updir)
		new_x /= np.linalg.norm(new_x)
		new_y = np.cross(new_y, updir)
		new_y /= np.linalg.norm(new_y)
		R = np.hstack((new_x, new_y, new_z))

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


