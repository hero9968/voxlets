import os
import collections
import numpy as np
import scipy.io

#from skimage import filter
from multiprocessing import Pool
from multiprocessing import cpu_count
import itertools
import timeit
import patches
import socket
import traceback
import sys

import paths

number_views = 42 # how many rendered views there are of each object

# in an ideal world we wouldn't have this hardcoded path, but life is too short to do it properly
host_name = socket.gethostname()


class DepthFeatureEngine(object):
	'''
	A class for computing features (and objective depths) from a front and a back
	depth image.
	Takes care of loading images, sampling random points from the image, extracting
	patches and other features.
	Can also save these features to file or return them as an array. 
	Doesn't do any forest training or prediction
	'''

	#samples_per_image = 1000  # how many random samples to take
	#hww = 7	 # half window width for the extracted patch size
	#frontrender = [] # the front depth render
	# backrender = [] # the back depth render
	# edge_image = [] # the edge image of the thing
	# mask = [] # which pixels are occupied by something and which are empty?
	# modelname = [] # the name of this movel
	# view_idx = [] # integer view idx


	def __init__(self, modelname, view_idx):
		self.modelname = modelname
		self.view_idx = view_idx
		self.frontrender = self.load_frontrender(modelname, view_idx)
		self.backrender = self.load_backrender(modelname, view_idx)
		
		#self.samples_per_image = 1000
		self.hww = 7
		self.indices = []

		#self.patch_extractor = patches.PatchEngine(output_patch_hww=self.hww, input_patch_hww=self.hww, fixed_patch_size=False)
		self.patch_extractor = patches.LocalSpiderEngine(t=7, fixed_patch_size=False)
		self.patch_extractor.compute_angles_image(self.frontrender)

		self.spider_engine = patches.SpiderEngine(self.frontrender, distance_measure='geodesic')
		self.spider_engine.focal_length = 240.0/(np.tan(np.rad2deg(43.0/2.0))) / 2.0
		self.spider_engine.angles_image = self.patch_extractor.angles

	def load_frontrender(self, modelname, view_idx):
		fullpath = paths.base_path + 'basis_models/renders/' + modelname + '/depth_' + str(view_idx) + '.mat'
		frontrender = scipy.io.loadmat(fullpath)['depth']
		self.mask = self.extract_mask(frontrender)
		return frontrender

	def load_backrender(self, modelname, view_idx):
		fullpath = paths.base_path + 'basis_models/render_backface/' + modelname + '/depth_' + str(view_idx) + '.mat'
		backrender = scipy.io.loadmat(fullpath)['depth']
		return backrender

	def extract_mask(self, render):
		mask = ~np.isnan(render)
		return mask

	def sample_from_mask(self, num_samples=500):
		'''
		Samples 2D locations from the 2D binary mask.
		If num_samples == -1, returns all valid locations from mask, otherwise returns random sample.
		Does not return any points within border_size of edge of mask
		'''
		self.samples_per_image = num_samples
		border_size = self.hww

		#border size is the width at the edge of the mask from which we are not allowed to sample
		self.indices = np.array(np.nonzero(self.mask))
		
		#if self.indices.shape[1]:
			#scipy.io.savemat('mask.mat', dict(mask=self.mask))

		# removing indices which are too near the border
		# Should use np.logical_or.reduce(...) instead of this...
		#print self.indices
		to_remove = np.logical_or(np.logical_or(self.indices[0] < border_size, 
												self.indices[1] < border_size),
								np.logical_or(self.indices[0] > self.mask.shape[0]-border_size,
												self.indices[1] > self.mask.shape[1]-border_size))
		self.indices = np.delete(self.indices, np.nonzero(to_remove), 1).transpose()

		# doing the sampling
		#print self.indices
		if self.samples_per_image == -1:
			pass # do nothing - keep all indices
		elif np.any(self.indices): # downsample indices
			samples = np.random.randint(0, self.indices.shape[0], self.samples_per_image)
			self.indices = self.indices[samples, :]
		else: # no indices - return empty list
			self.indices = []#np.tile(np.array(self.mask.shape)/2, (samples_per_image, 1))
		return self.indices

	def depth_difference(self, index):
		''' 
		returns the difference in depth between the front and the back
		renders at the specified (i, j) index
		'''
		return self.backrender[index[0], index[1]] - self.frontrender[index[0], index[1]]


	def compute_features_and_depths(self, verbose=False, jobs=1):

		# getting the mask from the points
		if verbose:
			print "View: " + str(self.view_idx) + " ... " + str(np.sum(np.sum(self.mask - self.extract_mask(self.backrender))))

		if not np.any(self.indices):
			self.patch_features = -np.ones((self.samples_per_image, 2*self.patch_extractor.output_patch_hww))
			self.spider_features = [-np.ones((1, 8)) for i in range(self.samples_per_image)]
			self.depth_diffs = [-1 for i in range(self.samples_per_image)]
			self.depths = [-1 for i in range(self.samples_per_image)]
			self.indices = [(-1, -1) for i in range(self.samples_per_image)]

		else:
			self.depths = [self.frontrender[index[0], index[1]] for index in self.indices]
			self.patch_features = self.patch_extractor.extract_patches(self.frontrender, self.indices)

			if jobs==1:
				#assert np.all(mask==extract_mask(backrender))
				self.spider_features = [self.spider_engine.compute_spider_feature(index) for index in self.indices]
				self.depth_diffs = [self.depth_difference(index) for index in self.indices]

			else:
				if jobs==-1: jobs = cpu_count()
				print "Multicore..." + str(jobs)
				pool = Pool(processes=jobs)
				self.spider_features = pool.map(self.spider_engine.compute_spider_feature, self.indices)
				self.depth_diffs = pool.map(self.depth_difference, self.indices)
				print "Done multicore..."


		num_samples = self.indices.shape[0]
		self.views = [self.view_idx for i in range(num_samples)]
		self.modelnames = [self.modelname for i in range(num_samples)]

	def features_and_depths_as_dict(self):
		'''
		Returns all the features and all the depths as a dict.
		Assumes they have all already been computed
		'''
		#print "Done view " + str(view_idx)
		return dict(patch_features=self.patch_features, 
					depth_diffs=self.depth_diffs, 
					indices=self.indices, 
					spider_features=self.spider_features, 
					depths=self.depths, 
					view_idxs=self.views, 
					modelnames=self.modelnames)

	def plot_index_samples(self, filename=[]):
		'''
		plot the front render and the sampled pairs
		'''
		import matplotlib.pyplot as plt
		plt.imshow(self.frontrender)
		plt.hold(True)
		plt.plot(self.indices[:, 1], self.indices[:, 0], '.')
		plt.hold(False)
		if filename:
			plt.savefig(filename)
		else:
			plt.show()


#features, depths, indices = zip(*(features_and_depths(modelname, view+1) 
						#for view in range(number_views)))

samples_per_image = 500

def compute_features(modelname_and_view):
	'''
	helper function to deal with the two-way problem
	'''
	engine = DepthFeatureEngine(modelname_and_view[0], modelname_and_view[1]+1)
	engine.sample_from_mask(samples_per_image)
	engine.compute_features_and_depths()
	return engine.features_and_depths_as_dict()


def list_of_dicts_to_dict(list_of_dicts):
	'''
	http://stackoverflow.com/questions/11450575/
	how-do-i-convert-a-list-of-dictionaries-to-a-dictionary-of-lists-in-python
	'''
	result = collections.defaultdict(list)

	for d in list_of_dicts:
		for k, v in d.items():
			result[k].append(v)

	return result

if host_name == 'troll':
	saving = True
	redo_if_exist = False # i.e. overwrite
	just_one = False
	multicore = True
else:
	saving = False
	redo_if_exist = True # i.e. overwrite
	just_one = True
	multicore = False

if __name__ == '__main__':

	if multicore:
		
		if host_name == 'troll':
			pool = Pool(processes=8)
		else:
			pool = Pool(processes=2)

	f = open(paths.models_list, 'r')

	for idx, line in enumerate(f):

		modelname = line.strip()
		#if not host_name == 'troll':
			#modelname = '12bfa757452ae83d4c5341ee07f41676'

		fileout = base_path + 'structured/features_nopatch/' + modelname + '.mat'

		if not redo_if_exist:
			if os.path.isfile(fileout): 
				temp = scipy.io.loadmat(fileout)['depths']
				if len(temp) == number_views * samples_per_image:
					print "Continuing model " + modelname
					continue

		print "Doing model " + modelname

		tic = timeit.default_timer()

		zipped_arguments = itertools.izip(itertools.repeat(modelname), range(number_views))

		if multicore:
			try:
				dict_list = pool.map(compute_features, zipped_arguments)
			except:
				print "Failed!!"
				print '-'*60
				traceback.print_exc(file=sys.stdout)
				print '-'*60
				continue
		else:
			dict_list = [compute_features(tt) for tt in zipped_arguments]

		fulldict = list_of_dicts_to_dict(dict_list)

		# reshaping the outputs to the corrct size
		for k, v in fulldict.items():
			#print k, np.array(v).shape
			fulldict[k] = np.array(v).reshape(number_views*samples_per_image, -1)

		# saving to file
		if saving:
			scipy.io.savemat(fileout, fulldict)
		else:
			print "WARNING: Not saving"

		print 'Done ' + str(idx) + ' in ' + str(timeit.default_timer() - tic)

		if just_one:
			break
		
	f.close()

#dict_list = []
#tic = timeit.default_timer()
#for view in range(number_views):
#	dict_list.append(features_and_depths(modelname, view+1))
#	print "Done view " + str(view)

# features = np.array(features).reshape(number_views*samples_per_image, -1)
# depths = np.array(depths).reshape(number_views*samples_per_image, 1)
# indices = np.array(indices).reshape(number_views*samples_per_image, 2)

# # now write the fulldict to a suitable mat file...
#fileout = base_path + 'features/' + modelname + '.mat'
#scipy.io.savemat('temp.mat', dict(features=features, depths=depths, indices=indices))
#scipy.io.savemat('temp.mat', fulldict)
