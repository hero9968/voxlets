'''
This is an engine for extracting rotated patches from a depth image.
Each patch is rotated so as to be aligned with the gradient in depth at that point
Patches can be extracted densely or from pre-determined locations
Patches should be able to vary to be constant-size in real-world coordinates
(However, perhaps this should be able to be turned off...)
'''

import numpy as np
import scipy.stats
import scipy.io
import cv2
from numbers import Number

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib as mpl


class CobwebEngine(object):
	'''
	A different type of patch engine, only looking at points in the compass directions
	'''

	def __init__(self, t, fixed_patch_size=False):

		# the stepsize at a depth of 1 m
		self.t = float(t)

		# dimension of side of patch in real world 3D coordinates
		#self.input_patch_hww = input_patch_hww

		# if fixed_patch_size is True:
		#   step is always t in input image pixels
		# else:
		#   step varies linearly with depth. t is the size of step at depth of 1.0 
		self.fixed_patch_size = fixed_patch_size 

	def set_image(self, im):
		self.im = im

	def get_cobweb(self, index):
		'''extracts cobweb for a single index point'''
		row, col = index
		
		start_angle = self.im.angles[row, col]
		start_depth = self.im.depth[row, col]

		if self.fixed_patch_size:
			offset_dist = self.t
		else:
			offset_dist = self.t / start_depth

		# computing all the offsets and angles efficiently
		offsets = offset_dist * np.array([1, 2, 3, 4])
		rad_angles = np.deg2rad(start_angle + np.array(range(0, 360, 45)))

		rows_to_take = (float(row) - np.outer(offsets, np.sin(rad_angles))).astype(int).flatten()
		cols_to_take = (float(col) + np.outer(offsets, np.cos(rad_angles))).astype(int).flatten()

		# defining the cobweb array ahead of time
		cobweb = np.nan * np.zeros((32, )).flatten()

		# working out which indices are within the image bounds
		to_use = np.logical_and.reduce((rows_to_take >= 0, 
										rows_to_take < self.im.depth.shape[0],
										cols_to_take >= 0,
										cols_to_take < self.im.depth.shape[1]))
		rows_to_take = rows_to_take[to_use]
		cols_to_take = cols_to_take[to_use]

		# computing the diff vals and slotting into the correct place in the cobweb feature
		vals = self.im.depth[rows_to_take, cols_to_take] - self.im.depth[row, col]
		cobweb[to_use] = vals
		return np.copy(cobweb.flatten())

	def extract_patches(self, indices):
		return [self.get_cobweb(index) for index in indices]


		#idxs = np.ravel_multi_index((rows_to_take, cols_to_take), dims=self.im.depth.shape, order='C')
		#cobweb = self.im.depth.take(idxs) - self.im.depth[row, col]


class SpiderEngine(object):
	'''
	Engine for computing the spider (compass) features
	'''

	def __init__(self, im):
		'''
		sets the depth image and computes the distance transform
		'''
		dt = DistanceTransforms(im)
		self.distance_transform = dt.get_compass_images()


	def compute_spider_features(self, idxs):
		'''
		computes the spider feature for a given point
		'''
		return self.distance_transform[idxs[0], idxs[1]]




class PatchPlot(object):
	'''
	Aim of this class is to plot boxes at specified locations, scales and orientations
	on a background image
	'''

	def __init__(self):
		pass

	def set_image(self, image):
		self.im = im
		plt.imshow(im.depth)

	def plot_patch(self, index, angle, width):
		'''plots a single patch'''
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


	def plot_patches(self, indices, scale_factor):
		'''plots the patches on the image'''
		
		scales = [scale_factor * self.im.depth[index[0], index[1]] for index in indices]
		
		angles = [self.im.angles[index[0], index[1]] for index in indices]

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

	import images
	import paths

	# loading the render
	im = images.CADRender()
	im.load_from_cad_set(paths.modelnames[30], 30)
	im.compute_edges_and_angles()

	# sampling indices from the image
	indices = np.array(np.nonzero(~np.isnan(im.depth))).transpose()
	samples = np.random.randint(0, indices.shape[0], 20)
	indices = indices[samples, :]

	# plotting patch
	patch_plotter = PatchPlot()
	patch_plotter.set_image(im.depth)
	patch_plotter.plot_patches(indices, scale_factor=10)

