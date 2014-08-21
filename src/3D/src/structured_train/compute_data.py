import os
import collections
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
#from skimage import filter
from multiprocessing import Pool
import itertools
import timeit

number_views = 42 # how many rendered views there are of each object
base_path = os.path.expanduser("~/projects/shape_sharing/data/3D/basis_models/")
models_list = base_path + 'databaseFull/fields/models.txt'

class depth_feature_engine(object):
	'''
	A class for computing features (and objective depths) from a front and a back
	depth image.
	Takes care of loading images, sampling random points from the image, extracting
	patches and other features.
	Can also save these features to file or return them as an array. 
	Doesn't do any forest training or prediction
	'''

	samples_per_image = 1000  # how many random samples to take
	hww = 7	 # half window width for the extracted patch size
	frontrender = [] # the front depth render
	backrender = [] # the back depth render
	edge_image = [] # the edge image of the thing
	mask = [] # which pixels are occupied by something and which are empty?
	modelname = [] # the name of this movel
	view_idx = [] # integer view idx

	# in an ideal world we wouldn't have this hardcoded path, but life is too short to do it properly
	base_path = os.path.expanduser("~/projects/shape_sharing/data/3D/basis_models/")

	def __init__(self, modelname, view_idx):
		self.modelname = modelname
		self.view_idx = view_idx
		self.frontrender = self.load_frontrender(modelname, view_idx)
		self.backrender = self.load_backrender(modelname, view_idx)
		self.mask = self.extract_mask(self.frontrender)

	def load_frontrender(self, modelname, view_idx):
		fullpath = base_path + 'renders/' + modelname + '/depth_' + str(view_idx) + '.mat'
		frontrender = scipy.io.loadmat(fullpath)['depth']
		return frontrender

	def load_backrender(self, modelname, view_idx):
		fullpath = base_path + 'render_backface/' + modelname + '/depth_' + str(view_idx) + '.mat'
		backrender = scipy.io.loadmat(fullpath)['depth']

		# hacking the backrender to insert nans...
		t = np.nonzero(np.abs(backrender-0.1) < 0.0001)
		backrender[t[0], t[1]] = np.nan

		return backrender

	def extract_patch(self, image, x, y, hww):
		return np.copy(image[x-hww:x+hww+1,y-hww:y+hww+1])

	def calc_patch_feature(self, depth_image, index):
		patch = self.extract_patch(depth_image, index[0], index[1], self.hww)
		patch_feature = (patch - depth_image[index[0], index[1]]).flatten()
		# feature = np.concatenate(patch, spider)
		return patch_feature

	def extract_mask(self, render):
		mask = ~np.isnan(render)
		return mask

	def sample_from_mask(self, num_samples=1, border_size=0):
		'''
		Samples 2D locations from the 2D binary mask.
		If num_samples == -1, returns all valid locations from mask, otherwise returns random sample.
		Does not return any points within border_size of edge of mask
		'''
		#border size is the width at the edge of the mask from which we are not allowed to sample
		indices = np.array(np.nonzero(self.mask))
		if indices.shape[1]:
			scipy.io.savemat('mask.mat', dict(mask=self.mask))

		# removing indices which are too near the border
		# Should use np.logical_or.reduce(...) instead of this...
		to_remove = np.logical_or(np.logical_or(indices[0] < border_size, 
												indices[1] < border_size),
								np.logical_or(indices[0] > self.mask.shape[0]-border_size,
												indices[1] > self.mask.shape[1]-border_size))
		indices = np.delete(indices, np.nonzero(to_remove), 1)

		# doing the sampling
		if num_samples == -1:
			return indices.transpose()
		elif np.any(indices):
			return indices[:, np.random.randint(0, indices.shape[1], num_samples)].transpose()
		else:
			return np.tile(np.array(mask.shape)/2, (num_samples, 1))

	def depth_difference(self, index):
		''' 
		returns the difference in depth between the front and the back
		renders at the specified (i, j) index
		'''
		return self.backrender[index[0], index[1]] - self.frontrender[index[0], index[1]]

	def compute_depth_edges(self, threshold):
		'''
		Finds the edges of a depth image. Not for noisy images!
		'''
		# convert nans to 0
		local_depthimage = np.copy(self.frontrender)
		#print self.frontrender
		t = np.nonzero(np.isnan(local_depthimage))
		#print t
		local_depthimage[t[0], t[1]] = 0

		# get the gradient and threshold
		dx,dy = np.gradient(local_depthimage, 1)
		edge_image = np.array(np.sqrt(dx**2 + dy**2) > 0.1)

		#plt.imshow(edge_image)
		#plt.show() # actually, don't show, just save to foo.png
		return edge_image

	def findfirst(self, array):
		'''
		Returns index of first non-zero element in numpy array
		'''
		T = np.where(array>0)
		if T[0].any():
			return T[0][0]
		else:
			return np.nan

	def calc_spider_features(self, index):
		'''
		Computes the distance to the nearest non-zero edge in each compass direction
		'''
		# extracting vectors along each compass direction
		compass = []
		compass.append(self.edge_image[index[0]:0:-1, index[1]]) # N
		compass.append(self.edge_image[index[0]+1:, index[1]]) # S
		compass.append(self.edge_image[index[0], index[1]+1:]) # E
		compass.append(self.edge_image[index[0], index[1]:0:-1]) # W
		compass.append(np.diag(np.flipud(self.edge_image[:index[0]+1, index[1]+1:]))) # NE
		compass.append(np.diag(self.edge_image[index[0]+1:, index[1]+1:])) # SE
		compass.append(np.diag(np.fliplr(self.edge_image[index[0]+1:, :index[1]]))) # SW
		compass.append(np.diag(np.flipud(np.fliplr(self.edge_image[:index[0], :index[1]])))) # NW

		spider = [self.findfirst(vec) for vec in compass]
		return spider

	def compute_features_and_depths(self):

		#load_frontrender(modelname, view_idx)
		#load_backrender(modelname, view_idx)

		#compute_depth_edges(0.1)
		#scipy.io.savemat('de.mat', dict(edgeimage=edgeimage))

		# getting the mask from the points
		self.edge_image = self.compute_depth_edges(0.1)
 		print "View: " + str(self.view_idx) + " ... " + str(np.sum(np.sum(self.mask - self.extract_mask(self.backrender))))

		#
		#assert np.all(mask==extract_mask(backrender))

		# sample pairs of coordinates
		indices = self.sample_from_mask(self.samples_per_image, self.hww)

		self.spider_features = [self.calc_spider_features(index) for index in indices]
		self.patch_features = [self.calc_patch_feature(self.frontrender, index) for index in indices]
		self.depth_diffs = [self.depth_difference(index) for index in indices]
		self.depths = [self.frontrender[index[0], index[1]] for index in indices]
		self.views = [self.view_idx for index in indices]
		self.modelnames = [self.modelname for index in indices]
		self.indices = indices

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
		plt.imshow(frontrender)
		plt.hold(True)
		plt.plot(indices[:, 1], indices[:, 0], '.')
		plt.hold(False)
		if filename:
			plt.savefig(filename)
		else:
			plt.show()

	def plot_edges(self, filename=[]):
		'''
		plot the edge image and the index point to a file
		(Probably remove this function pretty soon, it doens't really do anything...)
		'''
		plt.imshow(edgeimage)
		if filename:
			plt.savefig(filename)
		else:
			plt.show()


#features, depths, indices = zip(*(features_and_depths(modelname, view+1) 
						#for view in range(number_views)))

def compute_features(modelname_and_view):
	'''
	helper function to deal with the two-way problem
	'''
	engine = depth_feature_engine(modelname_and_view[0], modelname_and_view[1]+1)
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

samples_per_image = 1000

if __name__ == '__main__':

	pool = Pool(processes=4)              # start 4 worker processes

	f = open(models_list, 'r')

	for idx, line in enumerate(f):

		modelname = line.strip()
		fileout = base_path + 'features/' + modelname + '.mat'
		
		if os.path.isfile(fileout): 
			temp = scipy.io.loadmat(fileout)['depths']
			if len(temp) == number_views * samples_per_image:
				print "Continuing model " + modelname
				continue

		print "Doing model " + modelname

		tic = timeit.default_timer()

		zipped_arguments = itertools.izip(itertools.repeat(modelname), range(number_views))
		# try:
		# 	dict_list = pool.map(compute_features, zipped_arguments)
		# except:
		# 	print "Failed!!"
		# 	continue
		dict_list = [compute_features(tt) for tt in zipped_arguments]

		fulldict = list_of_dicts_to_dict(dict_list)

		# reshaping the outputs to the corrct size
		for k, v in fulldict.items():
			fulldict[k] = np.array(v).reshape(number_views*samples_per_image, -1)

		# saving to file
		
		scipy.io.savemat(fileout, fulldict)

		print 'Done ' + str(idx) + ' in ' + str(timeit.default_timer() - tic)
		
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

