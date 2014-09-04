'''
This is an engine for extracting rotated patches from a depth image.
Each patch is rotated so as to be aligned with the gradient in depth at that point
Patches can be extracted densely or from pre-determined locations
Patches should be able to vary to be constant-size in real-world coordinates
(However, perhaps this should be able to be turned off...)
'''

import numpy as np
import scipy.stats
import cv2
from numbers import Number

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib as mpl

class PatchEngine(object):

	def __init__(self, output_patch_hww, patch_size, fixed_patch_size=True, interp_method=cv2.INTER_CUBIC):

		# the hww of the OUTPUT patches
		self.output_patch_hww = output_patch_hww 

		# dimension of side of patch in real world 3D coordinates
		self.patch_size = patch_size 

		# if fixed_patch_size is True:
		#	patch is always patch_sizexpatch_size in input image pixels
		# else:
		# 	patch size varies linearly with depth. patch_size is the size of patch at depth of 1.0 
		self.fixed_patch_size = fixed_patch_size 
		#self.focal_length = focal_length # used for extracting constant-size patches
		
		# how does the interpolation get done when rotating patch
		self.interp_method = interp_method


	def extract_aligned_patch(self, img_in, row, col, hww):
		row = int(row)
		col = int(col)
		hww = int(hww)
		return np.copy(img_in[row-hww:row+hww+1,col-hww:col+hww+1])


	def extract_rotated_patch(self, index, angle=-1):
		'''
		output_patch_width  - size of extracted patch in input image pixels
		d    - size of extracted patch in output pixels
		angle - rotation angle for the patchs
		'''
		row,col = index
		assert(~np.isnan(self.image_to_extract[row, col]))

		if angle == -1:
			angle = self.angles[row, col]
		assert(~np.isnan(angle))
		depth = self.depth_image[row, col]

		'''Getting the oversized patch'''
		# total width of the output patch in input image pixels
		output_patch_width = 2*self.output_patch_hww+1

		# hww of the initial patch to extract - must ensure it is big enough to be rotated then downsized
		oversized_hww = np.ceil(float(output_patch_width)/np.sqrt(2.0)) + 2
		oversized_patch = self.extract_aligned_patch(self.image_to_extract, row, col, oversized_hww)

		'''Rotating the oversized patch'''
		
		# this is the centre coordinates of the patch
		oversized_patch_centre = (float(oversized_hww) + 0.5, float(oversized_hww) + 0.5)

		scale_factor = 1
		M = cv2.getRotationMatrix2D(oversized_patch_centre, angle, scale_factor)
		rotated_patch = cv2.warpAffine(oversized_patch, M, oversized_patch.shape, flags=self.interp_method)

		'''Extracting the middle of this rotated patch'''
		final_patch = self.extract_aligned_patch(rotated_patch, oversized_hww, oversized_hww, self.output_patch_hww)

		assert(final_patch.shape[0] == output_patch_width)
		assert(final_patch.shape[1] == output_patch_width)

		final_patch -= final_patch[self.output_patch_hww, self.output_patch_hww]
		#print final_patch[self.output_patch_hww, self.output_patch_hww]

		if False:
			plt.subplot(2, 3, 1)
			plt.imshow(oversized_patch, interpolation='none')
			plt.subplot(2, 3, 2)
			plt.imshow(rotated_patch, interpolation='none')
			plt.subplot(2, 3, 3)
			plt.imshow(final_patch, interpolation='none')
			plt.show()

		#print "Final patch: " + str(final_patch[self.output_patch_hww+1, self.output_patch_hww+1])
		#print "Image pixel: " + str(self.image_to_extract[row, col])

		#np.testing.assert_almost_equal(final_patch[self.output_patch_hww+1, self.output_patch_hww+1], 
	#								   self.image_to_extract[row, col],
	#								   decimal=1)

		return final_patch

	def fill_in_nans(self, image_to_repair, desired_mask):
		'''
		Fills in nans in the image_to_repair, with an aim of getting its non-nan values
		to take the shape of the desired_mask. Is only possible to fill in a nan
		if it borders (8-neighbourhood) a non-nan values.
		Otherwise, an error is thrown
		'''
		nans_to_fill = np.logical_and(np.isnan(image_to_repair), desired_mask)

		# replace each nan value with the nanmedian value of its neighbours 
		for row, col in np.array(np.nonzero(nans_to_fill)).T:
			bordering_vals = self.extract_aligned_patch(image_to_repair, row, col, 2)
			image_to_repair[row, col] = scipy.stats.nanmedian(bordering_vals.flatten())

		# values may be remaining. These will be filled in with zeros
		remaining_nans_to_fill = np.logical_and(np.isnan(image_to_repair), desired_mask)
		image_to_repair[remaining_nans_to_fill] = 0

		return image_to_repair


	def compute_angles_image(self, depth_image):
		'''
		Computes the angle of orientation at each pixel on the input depth image.
		The depth image is specified explicitly when calling this function
		in case one wants to extract patches from a different image to the one
		which the angles are computed from
		'''
		Ix = cv2.Sobel(depth_image, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=3)
		Iy = cv2.Sobel(depth_image, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=3)

		# here recover values for Ix and Iy, so they have same non-nan locations as depth image
		mask = ~np.isnan(depth_image)
		Ix = self.fill_in_nans(Ix, mask)
		Iy = self.fill_in_nans(Iy, mask)

		self.angles = np.rad2deg(np.arctan2(Iy, Ix))
		np.testing.assert_equal(mask, ~np.isnan(self.angles))
		self.depth_image = depth_image

		return self.angles

	def set_angles_to_zero(self):
		'''
		Specifies an angles image which is all zero 
		'''
		self.angles = -1

	def extract_patches(self, image, indices):
		'''
		Extract patches at the indices locations
		Returns them as a flattened matrix
		indices is an n x 2 matrix, in which each row is a [row, col] pixel index
		Assumes self.angles has already been computed!
		'''
		# Todo - remove inliers too close to edge of image (or throw error)
		# 
		# 

		# setting the image to extract from
		self.image_to_extract = image

		if isinstance(self.angles, Number) and self.angles == -1:
			self.angles = np.zeros(image.shape)

		# extracting all the patches
		patches = [self.extract_rotated_patch(index) for index in indices]

		# subtracting the middle pixel
		#patches = [patch - patch[self.output_patch_hww+1, self.output_patch_hww+1] for patch in patches]

		# need to do some kind of flattening here and conversion to numpy
		patch_array = np.array(patches).reshape((len(patches), -1))
		#print "Patch array is size : " + str(patch_array.shape)
		return patch_array


class PatchPlot(object):
	'''
	Aim of this class is to plot boxes at specified locations, scales and orientations
	on a background image
	'''

	def __init__(self):
		pass

	def set_image(self, image):
		self.image = image
		plt.imshow(image)

	def plot_patch(self, index, angle, width):
		'''
		plots single patch
		'''
		row, col = index
		bottom_left = (col - width/2, row - width/2)
		angle_rad = np.deg2rad(angle)

		# creating patch
		#print bottom_left, width, angle
		p_handle = patches.Rectangle(bottom_left, width, width, color="red", alpha=1.0, edgecolor='r', fill=None)
		transform = mpl.transforms.Affine2D().rotate_around(col, row, angle_rad) + plt.gca().transData
		p_handle.set_transform(transform)

		# adding to current plot
		plt.gca().add_patch(p_handle)

		# plotting line from centre to the edge
		plt.plot([col, col + width * np.cos(angle_rad)], 
				 [row, row + width * np.sin(angle_rad)], 'r-')


	def plot_patches(self, indices, angles, scales):
		'''
		plots the patches on the image
		'''

		if ~isinstance(scales, Number):
			scales = [scales for index in indices]

		plt.hold(True)

		for index, angle, scale in zip(indices, angles, scales):
			self.plot_patch(index, angle, scale)

		plt.hold(False)
		plt.show()



# here should probably write some kind of testing routine
# where an image is loaded, rotated patches are extracted and the gradient of the rotated patches
# is shown to be all mostly close to zero

if __name__ == '__main__':

	'''testing the plotting'''

	# loading the frontrender
	import loadsave
	obj_list = loadsave.load_object_names('all')
	modelname = obj_list[3]
	view_idx = 17
	frontrender = loadsave.load_frontrender(modelname, view_idx)

	# setting up patch engine for computation of angles
	patch_engine = PatchEngine(frontrender, 6, -1, -1)
	patch_engine.compute_angles_image(frontrender)

	# sampling indices from the image
	indices = np.array(np.nonzero(~np.isnan(frontrender))).transpose()
	samples = np.random.randint(0, indices.shape[0], 100)
	indices = indices[samples, :]

	angles = [patch_engine.angles[index[0], index[1]] for index in indices]

	# 
	patch_plotter = PatchPlot()
	patch_plotter.set_image(frontrender)
	patch_plotter.plot_patches(indices, angles, scales=10)


	# def remove_nans(inputs):
	# 	return [a for a in inputs if ~np.isnan(a)]

	# print np.nansum(np.array(angles_nearest) - np.array(angles_cubic))
	# plt.subplot(121)
	# plt.hist(remove_nans(angles_nearest), 50)
	# plt.subplot(122)
	# plt.hist(remove_nans(angles_cubic), 50)
	# plt.show()


