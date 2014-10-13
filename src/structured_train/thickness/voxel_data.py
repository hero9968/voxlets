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


	def __init__(self, frontrender=None, back_predictions=None):
		#self.vol = np.zeros((vol_size[0], vol_size[1], vol_size[2]))
		self.focal_length = 304.6
		if not frontrender==None and not back_predictions==None:
			self.fill_voxels(frontrender, back_predictions)
		

	def get_start_depth(self):
		return np.nanmin(np.concatenate((self.frontrender.flatten(), self.back_predictions.flatten())))

	def get_end_depth(self):
		return np.nanmax(np.concatenate((self.frontrender.flatten(), self.back_predictions.flatten())))

	def set_renders(self, frontrender, back_predictions):
		self.frontrender = frontrender
		self.back_predictions = back_predictions
		self.vol = []		

	def fill_slice(self, slice_idx):
		'''
		similar to fill_voxels but only doing one slice at a time
		'''

		# finding the start and end depths...
		start_depth = self.get_start_depth()
		end_depth = self.get_end_depth()
		scale_factor = (start_depth+end_depth) / (2 *self.focal_length)

		# create the volume...
		depth_in_voxels = int(np.ceil((end_depth - start_depth) / scale_factor))
		thisslice = np.zeros((depth_in_voxels, self.frontrender.shape[1]))
		print "Depth in vocels is " + str(depth_in_voxels)
		print "Start depth is " + str(start_depth)
		number_trees = self.back_predictions.shape[0]

		#max_depth_so_far = 0
		#min_depth_so_far = 10000

		# to do 
		# 1) use indices of the mask 
		# 2) convert front and back render to voxel coords before loop
		# 3) check for inside and outside the volume before the loop

		# populate the volume with each depth prediction...
		row = self.frontrender[slice_idx]
		for colidx, front_val in enumerate(row):
			front_vox_depth = round((front_val - start_depth) / scale_factor)

			for tree_idx in range(number_trees):

				back_val = self.back_predictions[tree_idx, slice_idx, colidx]
				if ~np.isnan(front_val) and ~np.isnan(back_val):
					back_vox_depth = round((back_val - start_depth) / scale_factor)
					thisslice[int(front_vox_depth):int(back_vox_depth), colidx] += 1
					#if back_vox_depth > max_depth_so_far:
					    #print rowidx, colidx, front_vox_depth, back_vox_depth
					    #max_depth_so_far = back_vox_depth
					#                    if front_vox_depth < min_depth_so_far:
					#                       print rowidx, colidx, front_vox_depth, back_vox_depth
					#                      min_depth_so_far = front_vox_
		#self.thisslice = vol
		return thisslice


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

	def extract_slice(self, slice_idx, axis=0):
		''' 
		returns a horizontal slice from the voxels
		(todo - also allow for vertical slices)
		'''
		if axis==0:
			volume_slice = self.vol[slice_idx, :, :].transpose()
		elif axis==1:
			volume_slice = self.vol[:, slice_idx, :].transpose()
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


