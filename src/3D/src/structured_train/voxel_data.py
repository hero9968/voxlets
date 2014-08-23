'''
The idea here is to have a voxel class, which stores the prediction results.
Will happily construct the voxels from the front and back renders
In an ideal world, perhaps should inherit from a more generic voxel class
I haven't thought about this yet though...
'''

import cv2
import numpy as np

class scene_voxels:

	# attributes


	def __init__(self, frontrender, back_predictions):
		#self.vol = np.zeros((vol_size[0], vol_size[1], vol_size[2]))
		self.focal_length = 304.6
		self.fill_voxels(frontrender, back_predictions)
		return

	def get_start_depth(self):
		return np.nanmin(np.concatenate((self.frontrender.flatten(), self.back_predictions.flatten())))

	def get_end_depth(self):
		return np.nanmax(np.concatenate((self.frontrender.flatten(), self.back_predictions.flatten())))


	def fill_voxels(self, frontrender, back_predictions):
		'''
		Populates the voxel array from the nxm frontrender and the list of nxm back predictions
		'''
		self.frontrender = frontrender
		self.back_predictions = back_predictions

		# finding the start and end depths...
		start_depth = self.get_start_depth()
		end_depth = self.get_end_depth()

		scale_factor = (start_depth+end_depth) / (2 *self.focal_length)
		print "Scale is " + str(scale_factor)

		# create the volume...
		depth_in_voxels = int(np.ceil((end_depth - start_depth) / scale_factor))
		vol = np.zeros((frontrender.shape[0], frontrender.shape[1], depth_in_voxels))
		print "Depth in vocels is " + str(depth_in_voxels)
		print "Start depth is " + str(start_depth)
		number_trees = back_predictions.shape[0]

		max_depth_so_far = 0
		min_depth_so_far = 10000

		# to do 
		# 1) use indices of the mask 
		# 2) convert front and back render to voxel coords before loop
		# 3) check for inside and outside the volume before the loop

		# populate the volume with each depth prediction...
		for rowidx, row in enumerate(frontrender):
			for colidx, front_val in enumerate(row):
				front_vox_depth = round((front_val - start_depth) / scale_factor)

				for tree_idx in range(number_trees):

					back_val = back_predictions[tree_idx, rowidx, colidx]
					if ~np.isnan(front_val) and ~np.isnan(back_val):
						back_vox_depth = round((back_val - start_depth) / scale_factor)
						vol[rowidx, colidx, int(front_vox_depth):int(back_vox_depth)] += 1
						#if back_vox_depth > max_depth_so_far:
						    #print rowidx, colidx, front_vox_depth, back_vox_depth
						    #max_depth_so_far = back_vox_depth
						#                    if front_vox_depth < min_depth_so_far:
						#                       print rowidx, colidx, front_vox_depth, back_vox_depth
						#                      min_depth_so_far = front_vox_
		self.vol = vol
		#return vol

	def extract_slice(self, slice_idx):
		''' 
		returns a horizontal slice from the voxels
		(todo - also allow for vertical slices)
		'''
		volume_slice = self.vol[slice_idx, :, :].transpose()
		return volume_slice

	def extract_warped_slice(self, slice_idx, output_image_height=500, maxdepth=-1):
		'''
		returns a horizontal slice warped to resemble the real-world positions
		maxdepth is the maximum distance to consider in the real-world coordinates
		'''
		mindepth = self.get_start_depth()
		if maxdepth == -1:
			maxdepth = self.get_end_depth()
		scale_factor = output_image_height / (maxdepth - mindepth)

		# computing the perspective transform between voxel space and output image space
		h = self.vol.shape[2]
		w = self.vol.shape[1]
		pts1 = np.float32([[0,0],[w,0],[0,h],[w,h]])

		d1 = scale_factor * (mindepth * float(self.vol.shape[1])/2) / self.focal_length
		d2 = scale_factor * (maxdepth * float(self.vol.shape[1])/2) / self.focal_length
		pts2 = np.float32([[d2-d1,0], [d1+d2,0], [0,output_image_height], [d2*2,output_image_height]])

		M = cv2.getPerspectiveTransform(pts1,pts2)

		# extracting and warping slice
		output_size = (2*int(d2),int(output_image_height))
		volume_slice = self.extract_slice(slice_idx)

		warped_image = cv2.warpPerspective(volume_slice,M,output_size, borderValue=1)
		return warped_image

