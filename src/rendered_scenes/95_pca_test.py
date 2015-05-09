# import numpy as np
# import cPickle as pickle
# import scipy.io

# from sklearn.decomposition import RandomizedPCA
# from sklearn.cluster import MiniBatchKMeans

# import paths
# import parameters

# if parameters.small_sample:
#     print "WARNING: Just computing on a small sample"


# def pca_randomized(X_in, local_subsample_length, num_pca_dims):

#     # take subsample
#     rand_exs = np.sort(np.random.choice(
#         X_in.shape[0],
#         np.minimum(local_subsample_length, X_in.shape[0]),
#         replace=False))
#     X = X_in.take(rand_exs, 0)

#     pca = RandomizedPCA(n_components=num_pca_dims)
#     pca.fit(X)
#     return pca

# # save path (open here so if an error is thrown I can catch it early...)

# # initialise lists
# shoeboxes = []


# for count, sequence in enumerate(paths.RenderedData.train_sequence()):

#     print "Processing " + sequence['name']

#     # loading the data
#     loadpath = paths.RenderedData.voxlets_dict_data_path + \
#         sequence['name'] + '.mat'
#     print "Loading from " + loadpath

#     D = scipy.io.loadmat(loadpath)
#     if 'shoeboxes' in D:
#         shoeboxes.append(D['shoeboxes'].astype(np.float16))
#     else:
#         print D.keys()

#     if count > parameters.max_sequences:
#         print "SMALL SAMPLE: Stopping"
#         break

# def convert_to_np(in_arr):

#     np_all_sboxes = np.concatenate(in_arr, axis=0)
#     print "All sboxes shape is " + str(np_all_sboxes.shape)
#     np_all_sboxes[np.isnan(np_all_sboxes)] = np.nanmax(np_all_sboxes)
#     return np_all_sboxes

# training_sboxes = convert_to_np(shoeboxes[:400])
# testing_sboxes = convert_to_np(shoeboxes[401:])

# print "Training is ", training_sboxes.shape
# print "Testing is ", testing_sboxes.shape

# components = [25, 50, 75, 100]
# # subsample_lengths = [25000, 50000, 75000, 100000]
# subsample_lengths = [2500, 5000, 7500, 10000]

# for component in components:
#     for subsample_length in subsample_lengths:

#         pca = pca_randomized(testing_sboxes, subsample_lengths, component)

#         # now find the reconstruction error on another datasets
#         testing_approx = pca.inverse_transform(pca.transform(testing_sboxes))
#         error = np.linalg.norm(testing_approx - testing_sboxes)
#         print component, subsample_length, error
