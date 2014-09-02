'''
This is an engine for extracting rotated patches from a depth image.
Each patch is rotated so as to be aligned with the gradient in depth at that point
Patches can be extracted densely or from pre-determined locations
Patches should be able to vary to be constant-size in real-world coordinates
(However, perhaps this should be able to be turned off...)
'''

import numpy as np
import cv2
import matplotlib.pyplot as plt

class PatchEngine(object):

	def __init__(self, depth_image, output_patch_hww, patch_size, focal_length):
		self.depth_image = depth_image
		self.output_patch_hww = output_patch_hww # the hww of the OUTPUT patches
		self.patch_size = patch_size # size of patch in real world 3D coordinates
		self.focal_length = focal_length # used for extracting constant-size patches
		
		# some hard-coded constants
		self.interp_method = cv2.INTER_NEAREST


	def extract_aligned_patch(self, img_in, row, col, hww):
		return np.copy(img_in[row-hww:row+hww+1,col-hww:col+hww+1])

	def extract_rotated_patch(self, index, angle=-1):
		'''
		p_w  - size of extracted patch in input image pixels
		d    - size of extracted patch in output pixels
		angle - rotation angle for the patchs
		'''
		row,col = index

		if angle == -1:
			angle = self.angles[row, col]
		assert(~np.isnan(angle))

		'''Getting the initial patch'''
		# total width of the output patch in input image pixels
		p_w = 2*self.output_patch_hww+1  

		# hww of the initial patch to extract - must ensure it is big enough to be rotated then downsized
		t_on_two = np.ceil(float(p_w)/np.sqrt(2.0)) + 2
		temp_patch = self.extract_aligned_patch(self.image_to_extract, row, col, t_on_two)

		'''Rotating the subpatch'''
		
		# this is half the size of the patch
		patch_centre = (float(t_on_two) + 0.5, float(t_on_two) + 0.5)

		scale_factor = 1
		M = cv2.getRotationMatrix2D(patch_centre, angle, scale_factor)
		rotated_patch = cv2.warpAffine(temp_patch, M, temp_patch.shape, flags=self.interp_method)

		'''Extracting the middle of this rotated patch'''
		#hww = int(float(p_w)/2)
		p_centre = int(float(temp_patch.shape[0]) / 2)
		final_patch = self.extract_aligned_patch(rotated_patch, p_centre, p_centre, self.output_patch_hww)

		assert(final_patch.shape[0] == p_w)
		assert(final_patch.shape[1] == p_w)

		if False:
			plt.subplot(2, 3, 1)
			plt.imshow(temp_patch, interpolation='none')
			plt.subplot(2, 3, 2)
			plt.imshow(rotated_patch, interpolation='none')
			plt.subplot(2, 3, 3)
			plt.imshow(final_patch, interpolation='none')

		return final_patch

	def compute_angles_image(self, depth_image):
		'''
		Computes the angle of orientation at each pixel on the input depth image.
		The depth image is specified explicitly when calling this function
		in case one wants to extract patches from a different image to the one
		which the angles are computed from
		'''
		Ix = cv2.Sobel(depth_image, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=3)
		Iy = cv2.Sobel(depth_image, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=3)
		self.angles = np.rad2deg(np.arctan2(Iy, Ix))
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

		if self.angles == -1:
			self.angles = np.zeros(image.shape)

		# extracting all the patches
		patches = [self.extract_rotated_patch(index) for index in indices]

		# need to do some kind of flattening here and conversion to numpy
		patch_array = np.array(patches).reshape((len(patches), -1))
		print "Patch array is size : " + str(patch_array.shape)
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

		# creating patch
		p_handle = patches.Rectangle(bottom_left, width, width, color="blue", alpha=0.0)
		transform = mpl.transforms.Affine2D().rotate_deg(angle) + plt.gca().transData
		p_handle.set_transform(transform)

		# adding to current plot
		plt.gca().add_patch(p_handle)

		# plotting line from centre to the edge
		plt.plot([col, col + width * np.cos(angle)], 
				 [row, row + width * np.sin(angle)], 'ro')


	def plot_patches(self, indices, angles, scales):
		'''
		plots the patches on the image
		'''

		plt.hold(True)
		for index, angle, scale in zip(indices, angles, scales):
			plot_patch(index, angle, scale)

		plt.hold(False)




# here should probably write some kind of testing routine
# where an image is loaded, rotated patches are extracted and the gradient of the rotated patches
# is shown to be all mostly close to zero

if __name__ == '__main__':

	def remove_nans(inputs):
		return [a for a in inputs if ~np.isnan(a)]

	print np.nansum(np.array(angles_nearest) - np.array(angles_cubic))
	plt.subplot(121)
	plt.hist(remove_nans(angles_nearest), 50)
	plt.subplot(122)
	plt.hist(remove_nans(angles_cubic), 50)
	plt.show()


