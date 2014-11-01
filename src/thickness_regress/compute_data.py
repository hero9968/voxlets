import os
import collections
import numpy as np
import scipy.io

#from skimage import filter
from multiprocessing import Pool
from multiprocessing import cpu_count
import itertools
import timeit
import socket
import traceback
import sys

sys.path.append(os.path.expanduser('~/project/shape_sharing/src/'))

from common import paths
from common import images
from common import features

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

	def __init__(self):
		self.indices = []
		self.im = []
		self.patch_extractor = features.CobwebEngine(t=5, fixed_patch_size=False)


	def random_sample_from_mask(self, num_samples=500):
		'''sample random points from the mask'''
		self.num_samples = num_samples
		#border size is the width at the edge of the mask from which we are not allowed to sample
		self.indices = np.array(np.nonzero(self.im.mask)).T

		if np.any(self.indices): # downsample indices
			samples = np.random.randint(0, self.indices.shape[0], self.num_samples)
			self.indices = self.indices[samples, :]
		else: # no indices - return empty list
			self.indices = []

		return self.indices


	def dense_sample_from_mask(self):
		'''samples all points from mask'''
		self.indices = np.array(np.nonzero(self.im.mask)).T

		self.num_samples = self.indices.shape[0]
		return self.indices


	def sample_numbered_slice_from_mask(self, slice_idx):
		'''samples the specified slice from the mask'''
		print np.nonzero(self.im.mask)[0]
		print np.nonzero(self.im.mask)[1]
		all_idxs = np.array(np.nonzero(self.im.mask)).T
		self.indices = np.array([[t0, t1] for t0, t1 in all_idxs if t0==slice_idx])
		print self.indices.shape

		self.num_samples = self.indices.shape[0]
		return self.indices


	def set_image(self, im):
		self.im = im

		# must update spider engine here as on initialisation it computes the distance transform
		self.spider_engine = features.SpiderEngine(im)
		self.patch_extractor.set_image(self.im)


	def compute_features_and_depths(self, verbose=False, jobs=1, features='all'):
		'''sets up the feature engines and computes the features'''

		if verbose:
			print "View: " + str(self.view_idx) + " ... " + str(np.sum(np.sum(self.im.mask - self.extract_mask(self.backrender))))

		# todo - move this rubbish into the feature engines...
		if not np.any(self.indices):
			raise Exception("output_patch_hww no longer exists")
			if features=='all' or 'cobweb' in features:
				self.patch_features = -np.ones((self.num_samples, 2*self.patch_extractor.output_patch_hww))
			if features=='all' or 'spider' in features:
				self.spider_features = [-np.ones((1, 8)) for i in range(self.num_samples)]
			self.depth_diffs = [-1 for i in range(self.num_samples)]
			self.depths = [-1 for i in range(self.num_samples)]
			self.indices = [(-1, -1) for i in range(self.num_samples)]

		else:
			self.depths = [self.im.depth[index[0], index[1]] for index in self.indices]
			if features=='all' or 'cobweb' in features:
				self.patch_features = self.patch_extractor.extract_patches(self.indices)
			else:
				self.patch_features =[]
			if features=='all' or 'spider' in features:
				self.spider_features = self.spider_engine.compute_spider_features(self.indices)
			else:
				self.spider_features =[]
			self.depth_diffs = [self.im.depth_difference(index) for index in self.indices]

			# if jobs==1:
				#assert np.all(mask==extract_mask(backrender))

			# else:
			# 	if jobs==-1: jobs = cpu_count()
			# 	print "Multicore..." + str(jobs)
			# 	pool = Pool(processes=jobs)
			# 	self.spider_features = pool.map(self.spider_engine.compute_spider_feature, self.indices)
			# 	print "Done multicore..."

		num_samples = self.indices.shape[0]
		self.views = [self.im.view_idx for i in range(self.num_samples)]
		self.modelnames = [self.im.modelname for i in range(self.num_samples)]


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
		plt.imshow(self.im.depth)
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
	if paths.data_type == 'cad':
		im = images.CADRender()
		im.load_from_cad_set(modelname_and_view[0], modelname_and_view[1]+1)
	elif paths.data_type == 'bigbird':
		im = images.CroppedRGBD()
		im.load_bigbird_from_mat(modelname_and_view[0], modelname_and_view[1])

	engine = DepthFeatureEngine()
	engine.set_image(im)
	engine.random_sample_from_mask(samples_per_image)
	engine.compute_features_and_depths()

	print "Done " + str(modelname_and_view[1])
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
	saving = True
	redo_if_exist = False # i.e. overwrite
	just_one = False
	multicore = False

if __name__ == '__main__':

	if multicore:
		
		if host_name == 'troll':
			pool = Pool(processes=10)
		else:
			pool = Pool(processes=2)

	f = open(paths.models_list, 'r')

	for idx, line in enumerate(f):

		modelname = line.strip()
		number_views = len(paths.views)
		#if not host_name == 'troll':
			#modelname = '12bfa757452ae83d4c5341ee07f41676'

		fileout = paths.feature_path + modelname + '.mat'

		if not redo_if_exist:
			if os.path.isfile(fileout): 
				temp = scipy.io.loadmat(fileout)['depths']
				if len(temp) == number_views * samples_per_image:
					print "Continuing model " + modelname
					continue

		print "Doing model " + modelname

		tic = timeit.default_timer()

		zipped_arguments = itertools.izip(itertools.repeat(modelname), 
										  paths.views)

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

