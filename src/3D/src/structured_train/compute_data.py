import os
import collections
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from skimage import filter
#import yaml
from multiprocessing import Pool
import itertools
import timeit

samples_per_image = 1000
hww = 7
number_views = 42

base_path = os.path.expanduser("~/projects/shape_sharing/data/3D/basis_models/")
models_list = base_path + 'databaseFull/fields/models.txt'

def load_frontrender(modelname, view_idx):
	fullpath = base_path + 'renders/' + modelname + '/depth_' + str(view_idx) + '.mat'
	return scipy.io.loadmat(fullpath)['depth']

def load_backrender(modelname, view_idx):
	fullpath = base_path + 'render_backface/' + modelname + '/depth_' + str(view_idx) + '.mat'
	backrender = scipy.io.loadmat(fullpath)['depth']

	# hacking the backrender to insert nans...
	t = np.nonzero(np.abs(backrender-0.1) < 0.0001)
	backrender[t[0], t[1]] = np.nan

	return backrender

def extract_patch(image, x, y, hww):
	return np.copy(image[x-hww:x+hww+1,y-hww:y+hww+1])

def extract_feature(depth_image, index):
	patch = extract_patch(depth_image, index[0], index[1], hww)
	patch_feature = (patch - depth_image[index[0], index[1]]).flatten()
	# feature = np.concatenate(patch, spider)
	return patch_feature

def extract_mask(render):
	return ~np.isnan(render)

def sample_from_mask(mask, num_samples=1, border_size=0):
	#border size is the width at the edge of the mask from which we are not allowed to sample
	indices = np.array(np.nonzero(mask))
	if indices.shape[1]:
		scipy.io.savemat('mask.mat', dict(mask=mask))

	# removing indices which are too near the border
	# Should use np.logical_or.reduce(...) instead of this...
	to_remove = np.logical_or(np.logical_or(indices[0] < border_size, 
											indices[1] < border_size),
							np.logical_or(indices[0] > mask.shape[0]-border_size,
											indices[1] > mask.shape[1]-border_size))
	indices = np.delete(indices, np.nonzero(to_remove), 1)

	# doing the sampling
	if np.any(indices):
		return indices[:, np.random.randint(0, indices.shape[1], num_samples)].transpose()
	else:
		return np.tile(np.array(mask.shape)/2, (num_samples, 1))

def depth_difference(backrender, frontrender, index):
	return backrender[index[0], index[1]] - frontrender[index[0], index[1]]

def depth_edges(depthimage, threshold):
	'''
	Finds the edges of a depth image. Not for noisy images!
	'''
	# convert nans to 0
	local_depthimage = np.copy(depthimage)
	t = np.nonzero(np.isnan(local_depthimage))
	local_depthimage[t[0], t[1]] = 0

	# get the gradient and threshold
	dx,dy = np.gradient(local_depthimage, 1)
	edge_image = np.sqrt(dx**2 + dy**2) > 0.1

	#plt.imshow(edge_image)
	#plt.show() # actually, don't show, just save to foo.png
	return edge_image

def findfirst(array):
	'''
	Returns index of first non-zero element in numpy array
	'''
	T = np.where(array>0)
	if T[0].any():
		return T[0][0]
	else:
		return np.nan


def spider_features(edgeimage, index):
	'''
	Computes the distance to the nearest non-zero edge in each compass direction
	'''
	#index = [100, 150];
	#print index

	# save the edge image and the index point to a file
	# plt.close('all')
	# plt.imshow(edgeimage)
	# plt.hold(True)
	# plt.plot(index[1], index[0], 'o')
	# plt.hold(False)
	# plt.savefig("edges.eps")
	# plt.close('all')

	# extracting vectors along each compass direction
	compass = []
	compass.append(edgeimage[index[0]:0:-1, index[1]]) # N
	compass.append(edgeimage[index[0]+1:, index[1]]) # S
	compass.append(edgeimage[index[0], index[1]+1:]) # E
	compass.append(edgeimage[index[0], index[1]:0:-1]) # W
	compass.append(np.diag(np.flipud(edgeimage[:index[0]+1, index[1]+1:]))) # NE
	compass.append(np.diag(edgeimage[index[0]+1:, index[1]+1:])) # SE
	compass.append(np.diag(np.fliplr(edgeimage[index[0]+1:, :index[1]]))) # SW
	compass.append(np.diag(np.flipud(np.fliplr(edgeimage[:index[0], :index[1]])))) # NW

	# f, (ax1, ax2, ax3, ax4) = plt.subplots(4)
	# ax1.imshow(compass[4])
	# ax2.imshow(compass[5])
	# ax3.imshow(compass[6])
	# ax4.imshow(compass[7])
	# plt.savefig("subplots.eps")

	spider = [findfirst(vec) for vec in compass]
	#print spider
	return spider

def features_and_depths(modelname, view_idx):
	frontrender = load_frontrender(modelname, view_idx)
	backrender = load_backrender(modelname, view_idx)

	edgeimage = depth_edges(frontrender, 0.1)
	#scipy.io.savemat('de.mat', dict(edgeimage=edgeimage))

	# getting the mask from the points
	mask = extract_mask(frontrender)
	print "View: " + str(view_idx) + " ... " + str(np.sum(np.sum(mask - extract_mask(backrender))))
	#assert np.all(mask==extract_mask(backrender))


	# sample pairs of coordinates
	indices = sample_from_mask(mask, samples_per_image, hww)

	# plot the front render and the sampled pairs
	if False:
		plt.imshow(frontrender)
		plt.hold(True)
		plt.plot(indices[:, 1], indices[:, 0], '.')
		plt.hold(False)
		plt.savefig("test_plot.png", dpi=96)

	spiders = [spider_features(edgeimage, index) 
						for index in indices]
	patch_features = [extract_feature(frontrender, index) 
						for index in indices]
	depth_diffs = [depth_difference(backrender, frontrender, index) 
						for index in indices]
	depths = [frontrender[index[0], index[1]] for index in indices]
	views = [view_idx for index in indices]
	modelnames = [modelname for index in indices]

	#print "Done view " + str(view_idx)
	return dict(patch_features=patch_features, depth_diffs=depth_diffs, 
		indices=indices, spider_features=spiders, depths=depths, 
		view_idxs=views, modelnames=modelnames)
	
	#return patch_features, depths, indices

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

#
#for line in f:
	#

# These will both be in loops ultimately!

#features, depths, indices = zip(*(features_and_depths(modelname, view+1) 
						#for view in range(number_views)))

def temp_features(modelname_and_view):
	'''
	helper function to deal with the two-way problem
	'''
	return features_and_depths(modelname_and_view[0], modelname_and_view[1]+1)


if __name__ == '__main__':

	pool = Pool(processes=4)              # start 4 worker processes

	f = open(models_list, 'r')

	for idx, line in enumerate(f):

		modelname = line.strip()
		fileout = base_path + 'features/' + modelname + '.mat'
		if os.path.isfile(fileout): continue
		print "Doing model " + modelname

		tic = timeit.default_timer()

		zipped_arguments = itertools.izip(itertools.repeat(modelname), range(number_views))
		try:
			dict_list = pool.map(temp_features, zipped_arguments)
		except:
			print "Failed!!"
			continue
		#dict_list = [temp_features(tt) for tt in zipped_arguments]

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

