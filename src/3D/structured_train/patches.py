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
#import arraypad # future function from numpy...

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib as mpl


class LocalSpiderEngine(object):
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

	def set_depth_image(self, depth_image):

		self.depth_image = depth_image
		self.compute_angles_image(depth_image)

	def get_offset_depth(self, start_row, start_col, angle_rad, offset):

		end_row = int(start_row - offset * np.sin(angle_rad))
		end_col = int(start_col + offset * np.cos(angle_rad))

		if end_row < 0 or end_col < 0 or \
			end_row >= self.depth_image.shape[0] or end_col >= self.depth_image.shape[1]:
			return np.nan
		else:
			return self.depth_image[end_row, end_col]

	def get_spider(self, index):
		'''
		'''
		row, col = index
		
		start_angle = self.angles[row, col]
		start_depth = self.depth_image[row, col]

		row = float(row)
		col = float(col)

		if self.fixed_patch_size:
			offset_dist = self.t
		else:
			offset_dist = self.t / start_depth

		spider = []
		for multiplier in [1, 2, 3, 4]:
			for offset_angle in range(0, 360, 45):
				offset_depth = self.get_offset_depth(row, col, np.deg2rad(start_angle + offset_angle), 
													offset_dist * multiplier)
				spider.append(offset_depth-start_depth)

		return spider

	def extract_patches(self, depth_image, indices):

		self.depth_image = depth_image
		print indices.shape

		return [self.get_spider(index) for index in indices]


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
			bordering_vals = self.extract_aligned_patch(image_to_repair, row, col, 2, np.nan)
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
		#Ix1 = np.copy(Ix)
		#Iy1 = np.copy(Iy)

		# here recover values for Ix and Iy, so they have same non-nan locations as depth image
		mask = 1-np.isnan(depth_image).astype(int)
		#scipy.io.savemat("output2.mat", dict(mask=mask, render=depth_image))
		#raise Exception("Break")

		Ix = self.fill_in_nans(Ix, mask)
		Iy = self.fill_in_nans(Iy, mask)

		self.angles = np.rad2deg(np.arctan2(Iy, Ix))

		mask_angles = np.logical_and(mask, np.isnan(self.angles))
		self.angles[mask_angles] = 0
		self.angles[mask==0] = np.nan

		angle_notnan = (~np.isnan(self.angles)).astype(int)

		try:
			np.testing.assert_equal(mask, angle_notnan)
		except:
			print np.sum(mask - ~np.isnan(self.angles))
			#import pdb; pdb.set_trace()
			#scipy.io.savemat("output.mat", dict(mask=mask, angles=self.angles, render=depth_image, Ix=Ix1, Iy=Iy1))

		self.depth_image = depth_image

		return self.angles

	def set_angles_to_zero(self):
		'''
		Specifies an angles image which is all zero 
		'''
		self.angles = -1


	def extract_aligned_patch(self, img_in, row, col, hww, pad_value=[]):
		top = int(row - hww)
		bottom = int(row + hww + 1)
		left = int(col - hww)
		right = int(col + hww + 1)

		im_patch = img_in[top:bottom, left:right]

		if top < 0 or left < 0 or bottom > img_in.shape[0] or right > img_in.shape[1]:
			if not pad_value:
				raise Exception("Patch out of range and no pad value specified")

			pad_left = int(self.negative_part(left))
			pad_top = int(self.negative_part(top))
			pad_right = int(self.negative_part(img_in.shape[1] - right))
			pad_bottom = int(self.negative_part(img_in.shape[0] - bottom))
			#print "1: " + str(im_patch.shape)

			im_patch = self.pad(im_patch, (pad_top, pad_bottom), (pad_left, pad_right),
								constant_values=100)
			#print "2: " + str(im_patch.shape)
			im_patch[im_patch==100.0] =pad_value # hack as apparently can't use np.nan as constant value

		if not (im_patch.shape[0] == 2*hww+1):
			print im_patch.shape
			print pad_left, pad_top, pad_right, pad_bottom
			print left, top, right, bottom
			im_patch = np.zeros((2*hww+1, 2*hww+1))
		if not (im_patch.shape[1] == 2*hww+1):
			print im_patch.shape
			print pad_left, pad_top, pad_right, pad_bottom
			print left, top, right, bottom
			im_patch = np.zeros((2*hww+1, 2*hww+1))
			#raise Exception("Bad shape!")

		return np.copy(im_patch)




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
		self.angles_image = np.rad2deg(np.arctan2(dy, dx))

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

		if np.isnan(angle):
			angle = 0
			print "Warning! Angle was nan in line extraction"

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


	def in_range(self, point_2d, border=0):
		'''
		returns true if point_2d (col, row) is inside the depth image, false Otherwise
		'''
		return point_2d[0] >= border and \
				point_2d[1] >= border and \
				point_2d[0] < (self.depthimage.shape[1]-border) and \
				point_2d[1] < (self.depthimage.shape[0]-border)

	def get_spider_distance(self, start_point, angle_deg):
		'''
		returns the distance to the nearest depth edge 
		along the line with specified angle, starting at start_point

		TODO - deal with edges of image - return some kind of -1 or nan...
		'''
		# get generator for all the points on the line
		angle = np.deg2rad(angle_deg%360)
		line_points = self.line(start_point, angle)
		
		if self.distance_measure == 'pixels' or self.distance_measure == 'perpendicular':

			# traverse line until we hit a point which is actually on the depth image
			for idx,line_point in enumerate(line_points):
				#self.blank_im[line_point[1], line_point[0]] = 1 # debugging
				if self.edge_image[line_point[1], line_point[0]]:# or not self.in_range(line_point):
					break

			distance = self.distance_2d(start_point, line_point)

			if self.distance_measure == 'perpendicular':
				distance *= self.depthimage[line_point[1], line_point[0]]

		elif self.distance_measure == 'geodesic':

			# loop until we hit a point which is actually on the depth image
			position_depths = [] # stores the x,y position and the depths of points along curve

			for idx, line_point in enumerate(line_points):

				#print "Doing pixel " + str(idx) + str(line_point)
				#self.blank_im[line_point[1], line_point[0]] = 1 # debugging
				end_of_line = not self.in_range(line_point, border=1) or self.edge_image[line_point[1], line_point[0]]

				# only look at every 10th point, for efficiency & noise robustness
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

		else:
			raise Exception("Unknown distance measure: " + self.distance_measure)

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

		# nip nans in the bud here!
		if np.isnan(self.depthimage[row, col]):
			#raise Exception("Nan depth!")
			print "Nan depth!"
			return [0 for i in range(8)]
		elif np.isnan(start_angle):
			print "Nan start angle!"
			return [0 for i in range(8)]
	
		return [self.get_spider_distance(start_point,  start_angle + offset_angle)
				for offset_angle in range(360, 0, -45)]



# here should probably write some kind of testing routine
# where an image is loaded, rotated patches are extracted and the gradient of the rotated patches
# is shown to be all mostly close to zero

if __name__ == '__main__':

	'''testing the plotting'''

	# loading the frontrender
	import loadsave
	obj_list = loadsave.load_object_names('all')
	modelname = obj_list[3]
	view_idx = 2
	frontrender = loadsave.load_frontrender(modelname, view_idx)

	# setting up patch engine for computation of angles
	patch_engine = PatchEngine(frontrender, 6, -1, -1)
	patch_engine.compute_angles_image(frontrender)

	# sampling indices from the image
	indices = np.array(np.nonzero(~np.isnan(frontrender))).transpose()
	samples = np.random.randint(0, indices.shape[0], 20)
	indices = indices[samples, :]

	angles = [patch_engine.angles[index[0], index[1]] for index in indices]
	depths = [patch_engine.depth_image[index[0], index[1]] for index in indices]
	print depths

	scales = (10*np.array(depths)).astype(int)
	print scales

	# 
	patch_plotter = PatchPlot()
	patch_plotter.set_image(frontrender)
	patch_plotter.plot_patches(indices, angles, scales=scales)


	# def remove_nans(inputs):
	# 	return [a for a in inputs if ~np.isnan(a)]

	# print np.nansum(np.array(angles_nearest) - np.array(angles_cubic))
	# plt.subplot(121)
	# plt.hist(remove_nans(angles_nearest), 50)
	# plt.subplot(122)
	# plt.hist(remove_nans(angles_cubic), 50)
	# plt.show()
