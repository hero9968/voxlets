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

	def __init__(self, output_patch_hww, input_patch_hww, fixed_patch_size=True, interp_method=cv2.INTER_CUBIC):

		# the hww of the OUTPUT patches
		self.output_patch_hww = output_patch_hww 

		# dimension of side of patch in real world 3D coordinates
		self.input_patch_hww = input_patch_hww

		# if fixed_patch_size is True:
		#	patch is always patch_sizexpatch_size in input image pixels
		# else:
		# 	patch size varies linearly with depth. patch_size is the size of patch at depth of 1.0 
		self.fixed_patch_size = fixed_patch_size 
		#self.focal_length = focal_length # used for extracting constant-size patches
		
		# how does the interpolation get done when rotating patch
		self.interp_method = interp_method

	def negative_part(self, num_in):
		return [0 if num_in >= 0 else np.abs(num_in)]

	def extract_aligned_patch(self, img_in, row, col, hww, pad_value=[]):
		top = int(row - hww)
		bottom = int(row + hww + 1)
		left = int(col - hww)
		right = int(col + hww + 1)

		im_patch = img_in[top:bottom, left:right]

		if top < 0 or left < 0 or bottom >= img_in.shape[0] or right >= img_in.shape[1]:

			if not pad_value:
				raise Exception("Patch out of range and no pad value specified")

			pad_left = self.negative_part(left)
			pad_top = self.negative_part(top)
			pad_right = self.negative_part(img_in.shape[1] - right)
			pad_bottom = self.negative_part(img_in.shape[0] - bottom)

			im_patch = np.pad(im_patch, ((pad_top, pad_bottom), (pad_left, pad_right)),
								mode='constant', constant_values=pad_value)

		assert(im_patch.shape[0] == 2*hww+1)
		assert(im_patch.shape[1] == 2*hww+1)

		return np.copy(im_patch)


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

		if self.fixed_patch_size:
			input_patch_hww = self.input_patch_hww
		else:
			input_patch_hww = int(float(self.input_patch_hww) * depth)
			#print depth, self.input_patch_hww, input_patch_hww


		# scale factor is how much to stretch the input patch by 
		scale_factor = float(self.output_patch_hww) / float(input_patch_hww)

		'''Getting the oversized patch'''

		# hww of the initial patch to extract - must ensure it is big enough to be rotated then downsized
		oversized_hww = int(np.sqrt(2.0) * input_patch_hww + 3)
		oversized_patch = self.extract_aligned_patch(self.image_to_extract, row, col, oversized_hww, pad_value=np.nan)
		#print oversized_patch.shape

		'''Rotating the oversized patch'''
		
		# this is the centre coordinates of the patch
		oversized_patch_centre = (float(oversized_hww) + 0.5, float(oversized_hww) + 0.5)

		M = cv2.getRotationMatrix2D(oversized_patch_centre, angle, scale_factor)
		try:
			rotated_patch = cv2.warpAffine(oversized_patch, M, oversized_patch.shape, flags=self.interp_method)
		except:
			print "o_hww = " + str(oversized_hww)
			print "depth = " + str(depth)
			print row,col
			print self.image_to_extract[row, col]
			print oversized_patch
			print M
			print oversized_patch.shape
			raise Exception("Cannot do the rotation")			

		'''Extracting the middle of this rotated patch'''
		final_patch = self.extract_aligned_patch(rotated_patch, oversized_hww, oversized_hww, self.output_patch_hww)

		assert(final_patch.shape[0] == 2*self.output_patch_hww+1)
		assert(final_patch.shape[1] == 2*self.output_patch_hww+1)

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

		if isinstance(scales, Number):
			scales = [scales * self.image[index[0], index[1]] for index in indices]

		plt.hold(True)

		for index, angle, scale in zip(indices, angles, scales):
			self.plot_patch(index, angle, scale)

		plt.hold(False)
		plt.show()




class SpiderEngine(object):
	'''
	Engine for computing the spider (compass) features
	'''

	def __init__(self, depthimage, edge_threshold=5, distance_measure='geodesic'):

		self.edge_threshold = edge_threshold

		self.dilation_parameter = 1
		self.set_depth_image(depthimage)

		# this should be 'geodesic', 'perpendicular' or 'pixels'
		self.distance_measure = distance_measure


	def set_depth_image(self, depthimage):
		'''
		saves the depth image, also comptues the edges and sets up
		the dictionary
		'''
		self.depthimage = depthimage
		self.compute_depth_edges()
		self.dilate_depth_edges(self.dilation_parameter)

	def dilate_depth_edges(self, dilation_size):
		'''
		Dilates the depth image.
		This is important to ensure the spider lines definitely 
		touch the edge
		'''
		kernel = np.ones((dilation_size, dilation_size),np.uint8)
		self.depthimage = cv2.dilate(self.depthimage, kernel, iterations=1)

	def compute_depth_edges(self):
		'''
		sets and returns the edges of a depth image. 
		Not good for noisy images!
		'''
		# convert nans to 0
		local_depthimage = np.copy(self.depthimage)
		local_depthimage[np.isnan(local_depthimage)] = 0

		# get the gradient and threshold
		dx,dy = np.gradient(local_depthimage, 1)
		self.edge_image = np.array(np.sqrt(dx**2 + dy**2) > 0.1)

		return self.edge_image

	def line(self, start_point, angle):
		'''
		Bresenham's line algorithm
		(Adapted from the internet)
		Uses Yield for efficiency, as we probably don't need to generate 
		all the points along the line!
		Additionally returns the stepsize between the previous and current
		point. This is useful for computing a geodesic distance
		'''
		MAX_POINT = 10000 # to prevent infinite loops...

		x0, y0 = start_point
		x, y = x0, y0
		stepsize = 0 # squared size of step between previous point and current point

		dx = int(abs(1000 * np.cos(angle)))
		dy = int(abs(1000 * np.sin(angle)))

		sx = -1 if np.cos(angle) < 0 else 1
		sy = -1 if np.sin(angle) < 0 else 1

		if dx > dy:
			err = dx / 2 # changed 2.0 to 2
			while abs(x) != MAX_POINT:
				yield (x, y)
				err -= dy
				if err < 0:
					y += sy
					err += dx
				x += sx
			else:
				raise Exception("Exceeded maximum point...")

		else:
			err = dy / 2 # changes 2.0 to 2
			while abs(y) != MAX_POINT:
				yield (x, y)
				err -= dx
				if err < 0:
					x += sx
					err += dy
				y += sy
			else:
				raise Exception("Exceeded maximum point...")

		yield (x, y)
		

	def distance_2d(self, p1, p2):
		'''
		Euclidean distance between two points in 2d
		'''
		return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

	def distance_3d(self, p1, p2):
		'''
		Euclidean distance between two points in 2d
		'''
		return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2 + (p1[2] - p2[2])**2)


	def in_range(self, point_2d):
		'''
		returns true if point_2d (col, row) is inside the depth image, false Otherwise
		'''
		return point_2d[0] >= 0 and \
				point_2d[1] >= 0 and \
				point_2d[0] <= self.depthimage.shape[1] and \
				point_2d[1] <= self.depthimage.shape[0]

	def get_spider_distance(self, start_point, angle_deg):
		'''
		returns the distance to the nearest depth edge 
		along the line with specified angle, starting at start_point

		TODO - deal with edges of image - return some kind of -1 or nan...
		'''
		# get generator for all the points on the line
		angle = np.deg2rad(angle_deg%360)
		#print
		line_points = self.line(start_point, angle)
		
		if self.distance_measure == 'pixels' or self.distance_measure == 'perpendicular':

			# traverse line until we hit a point which is actually on the depth image
			for idx,line_point in enumerate(line_points):
				self.blank_im[line_point[1], line_point[0]] = 1
				if self.edge_image[line_point[1], line_point[0]]:# or not self.in_range(line_point):
					break

			distance = self.distance_2d(start_point, line_point)

			#if self.distance_measure == 'perpendicular':
				#distance *= self.depthimage[line_point[1], line_point[0]]

		elif self.distance_measure == 'geodesic':

			# loop until we hit a point which is actually on the depth image
			position_depths = [] # stores the x,y position and the depths of points along curve
			for idx, line_point in enumerate(line_points):

				#print "Doing pixel " + str(idx) + str(line_point)
				self.blank_im[line_point[1], line_point[0]] = 1
				end_of_line = self.edge_image[line_point[1], line_point[0]] or not self.in_range(line_point)

				if (idx % 10) == 0 or end_of_line:
					current_depth = self.depthimage[line_point[1], line_point[0]]
					position_depths.append((line_point, current_depth))

				if end_of_line:
					break

			# adding up the geodesic distance for all the points
			distance = 0
			p1 = self.project_3d_point(position_depths[0][0], position_depths[0][1])
			for pos, depth in position_depths[1:]:
				p0 = p1
				p1 = self.project_3d_point(pos, depth)
				distance += self.distance_3d(p0, p1)

		return distance

	def project_3d_point(self, position, depth):
		'''
		returns the 3d point at (x, y) point 'position' and
		at depth 'depth'
		'''
		return [(position[0] - self.depthimage.shape[1]) * depth / self.focal_length,
				(position[1] - self.depthimage.shape[0]) * depth / self.focal_length,
				depth]

	def compute_spider_feature(self, index):
		'''
		computes the spider feature for a given point with a given starting angle
		'''

		row, col = index
		start_point = (col, row)
		start_angle = self.angles_image[row, col]
		self.blank_im = self.depthimage / 10.0
		#self.blank_im[self.blank_im==0] = np.nan
	
		return [self.get_spider_distance(start_point,  start_angle + offset_angle)
				for offset_angle in range(0, 360, 45)]




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


