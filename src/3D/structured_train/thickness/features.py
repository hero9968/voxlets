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

	def __init__(self, distance_measure='geodesic'):

		# this should be 'geodesic', 'perpendicular' or 'pixels'
		self.distance_measure = distance_measure


	def set_image(self, im):
		'''sets the depth image'''
		self.im = im

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
				point_2d[0] < (self.im.depth.shape[1]-border) and \
				point_2d[1] < (self.im.depth.shape[0]-border)

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
				distance *= self.im.depth[line_point[1], line_point[0]]

		elif self.distance_measure == 'geodesic':

			# loop until we hit a point which is actually on the depth image
			position_depths = [] # stores the x,y position and the depths of points along curve

			for idx, line_point in enumerate(line_points):

				#print "Doing pixel " + str(idx) + str(line_point)
				#self.blank_im[line_point[1], line_point[0]] = 1 # debugging
				end_of_line = not self.in_range(line_point, border=1) or self.im.edges[line_point[1], line_point[0]]

				# only look at every 10th point, for efficiency & noise robustness
				if (idx % 10) == 0 or end_of_line:
					current_depth = self.im.depth[line_point[1], line_point[0]]
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
		return [(position[0] - self.im.depth.shape[1]) * depth / self.im.focal_length,
				(position[1] - self.im.depth.shape[0]) * depth / self.im.focal_length,
				depth]

	def compute_spider_feature(self, index):
		'''
		computes the spider feature for a given point with a given starting angle
		'''
		row, col = index
		start_point = (col, row)
		start_angle = self.im.angles[row, col]
		self.blank_im = self.im.depth / 10.0

		# nip nans in the bud here!
		if np.isnan(self.im.depth[row, col]):
			#raise Exception("Nan depth!")
			print "Nan depth!"
			return [0 for i in range(8)]
		elif np.isnan(start_angle):
			print "Nan start angle!"
			return [0 for i in range(8)]
	
		return [self.get_spider_distance(start_point,  start_angle + offset_angle)
				for offset_angle in range(360, 0, -45)]




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



class DistanceTransforms(object):
	'''
	like the spider feature but aligned with dimensions
	'''
	def __init__(self, im=[]):
		self.set_im(im)


	def set_im(self, im):
		self.im = im


	def straight_dist_transform(self, direction):
		'''
		axis aligned distance transform, going from left to the right
		and top to bottom and vv
		'''

		if direction=='e':
			edge_im = self.im.edges
			ray_image = self.im.ray_image
			depth_image = self.im.depth
		elif direction=='w':
			edge_im = np.fliplr(self.im.edges)
			ray_image = np.fliplr(self.im.ray_image)
			depth_image = np.fliplr(self.im.depth)
		elif direction=='s':
			edge_im = self.im.edges.T
			ray_image = self.im.ray_image.T
			depth_image = self.im.depth.T
		elif direction=='n':
			edge_im = np.fliplr(self.im.edges.T)
			ray_image = np.fliplr(self.im.ray_image.T)
			depth_image = np.fliplr(self.im.depth.T)

		pixel_count_im = np.nan * np.copy(edge_im).astype(np.float)
		geodesic_im = np.nan * np.copy(edge_im).astype(np.float)

		u = np.arange(-edge_im.shape[1]/2, edge_im.shape[1]/2)

		# loop over each row...
		for row_idx, row in enumerate(edge_im):
			if np.any(row):

				dists = u*depth_image[row_idx]
				dist_diffs = np.abs(np.insert(np.diff(dists), 0, 0))

				pixel_count = np.nan
				geo_dist = np.nan
				previous_ray = 0

				for col_idx, pix in enumerate(row):
					
					if pix:
						pixel_count = 0
						geo_dist = 0
					else:
						pixel_count += 1
						geo_dist += dist_diffs[col_idx]

					pixel_count_im[row_idx, col_idx] = pixel_count
					geodesic_im[row_idx, col_idx] = geo_dist

		out_stack = np.dstack((pixel_count_im, geodesic_im))

		if direction=='w':
			out_stack = np.fliplr(out_stack)
		elif direction=='s':
			out_stack = np.transpose(out_stack, axes=[1, 0, 2])
		elif direction=='n':
			out_stack = np.transpose(np.fliplr(out_stack), axes=[1, 0, 2])

		return out_stack

#	def row_dists(self, edges_row, depth_row):




	def se_dist_transform(self, im):
		# create the output image
		out_im = np.nan * np.copy(im).astype(np.float)
			
		# pixels below the diagonal - loop over each row
		for row_idx in range(im.shape[0]):
			
			count = np.nan # keeps count of pix since last edge
			num_to_count = min(im.shape[1], im.shape[0] - row_idx)
			
			# now speed down the diagonal
			for row_col_counter in range(num_to_count):
				pix = im[row_idx + row_col_counter, row_col_counter]
				count = 0 if pix else count + 1
				out_im[row_idx + row_col_counter, row_col_counter] = count

		# pixels above the diagonal - loop over each column
		for col_idx in range(im.shape[1]):
			
			count = np.nan # keeps count of pix since last edge
			num_to_count = min(im.shape[0], im.shape[1] - col_idx)
			
			# now speed down the diagonal
			for row_col_counter in range(num_to_count):
				pix = im[row_col_counter, col_idx + row_col_counter]
				count = 0 if pix else count + 1
				out_im[row_col_counter, col_idx + row_col_counter] = count

		return np.dstack((out_im, out_im))


	def sw_dist_transform(self, im):
		return self.se_dist_transform(np.fliplr(im))


	def nw_dist_transform(self, im):
		return np.fliplr(self.ne_dist_transform(np.fliplr(im)))


	def ne_dist_transform(self, im):
		temp = self.sw_dist_transform(im.T)
		print temp.shape
		return np.transpose(temp, axes=(1, 0, 2))


	def get_compass_images(self):
		return [self.straight_dist_transform('n'),
				self.straight_dist_transform('e'),
				self.straight_dist_transform('s'),
				self.straight_dist_transform('w')]

				# self.se_dist_transform(self.im.edges),
				# self.s_dist_transform(self.im.edges, self.im.ray_image),
				# self.sw_dist_transform(self.im.edges),
				# self.w_dist_transform(self.im.edges, self.im.ray_image),
				# self.nw_dist_transform(self.im.edges),
				# self.n_dist_transform(self.im.edges, self.im.ray_image),
				# self.ne_dist_transform(self.im.edges)]




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

