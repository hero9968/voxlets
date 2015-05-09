'''
Things I'm not really using
'''


def render_diff_view(grid1, grid2, savepath):
    '''
    renders grid1, and any nodes not in grid2 get done in a different colour
    '''
    # convert nans to the minimum
    ms1 = mesh.Mesh()
    ms1.from_volume(grid1, 0)
    ms1.remove_nan_vertices()

    ms2 = mesh.Mesh()
    ms2.from_volume(grid2, 0)
    ms2.remove_nan_vertices()

    # now do a sort of setdiff between the two...
    print "Bulding dictionary", ms2.vertices.shape
    ms2_dict = {}
    for v in ms2.vertices:
        vt = (100*v).astype(int)
        ms2_dict[(vt[0], vt[1], vt[2])] = 1

    print "Done bulding dictionary", ms1.vertices.shape

    # label each vertex in ms1
    labels = np.zeros(ms1.vertices.shape[0])
    for count, v in enumerate(ms1.vertices):
        vt = (100*v).astype(int)
        if (vt[0], vt[1], vt[2]) in ms2_dict:
            labels[count] = 1
    print "Done checking dictionary"

    # memory problem?
    ms1.write_to_ply('/tmp/temp.ply', labels)

    sp.call([paths.blender_path,
             "../rendered_scenes/spinaround/spin.blend",
             "-b", "-P",
             "../rendered_scenes/spinaround/blender_spinaround_frame_ply.py"],
             stdout=open(os.devnull, 'w'),
             close_fds=True)

    #now copy file from /tmp/.png to the savepath...
    print "Moving render to " + savepath
    shutil.move('/tmp/.png', savepath)





class VoxelGridCollection(object):
    '''
    class for doing things to a list of same-sized voxelgrids
    Not ready to use yet - but might be good one day!
    '''
    def __init__(self):
        raise Exception("Not ready to use!")

    def set_voxelgrids(self, voxgrids_in):
        self.voxgrids = voxgrids_in

    def cluster_voxlets(self, num_clusters, subsample_length):

        '''helper function to cluster voxlets'''

        # convert to np array
        np_all_sboxes = np.concatenate(shoeboxes, axis=0)
        all_sboxes = np.array([sbox.V.flatten() for sbox in self.voxlist]).astype(np.float16)

        # take subsample
        if local_subsample_length > X.shape[0]:
            X_subset = X
        else:
            to_use_for_clustering = \
                np.random.randint(0, X.shape[0], size=(local_subsample_length))
            X_subset = X[to_use_for_clustering, :]

        # doing clustering
        km = MiniBatchKMeans(n_clusters=num_clusters)
        km.fit(X_subset)


        # if segment:
        #     # segmenting with just the visible voxels
        #     self.visible_labels = self._segment_tsdf_project_2d(
        #         self.im_tsdf, z_threshold=2, floor_height=4)

        #     # #### >> transfer the labels from the voxel grid to the image
        #     self.visible_im_label = self.im.label_from_grid(self.visible_labels)

        #     # expanding these labels to also cover all the unobserved regions
        #     uv, to_project_idxs = self.im_tsdf.project_unobserved_voxels(self.im)
        #     inside_image = self.im.find_points_inside_image(uv)

        #     # labels of all the non-nan voxels inside the image...
        #     vox_labels = self.visible_im_label[
        #         uv[inside_image, 1], uv[inside_image, 0]]

        #     # now propograte these labels back to the main grid
        #     temp = to_project_idxs[inside_image]
        #     self.visible_labels.V.ravel()[temp] = vox_labels
        #     # << ####

        #     print "Separate out the partial tsdf into different 'layers' in a grid..."

        #     self.visible_labels_separate = \
        #         self._separate_binary_grids(self.visible_labels.V, True)

        #     temp = self.im_tsdf.copy()
        #     temp.V[np.isnan(temp.V)] = np.nanmin(temp.V)

        #     self.visible_tsdf_separate = \
        #         self._label_grids_to_tsdf_grids(temp, self.visible_labels_separate)


        # # transfer the labels from the voxel grid tothe image
        # self.im.label_from_grid(self.gt_labels)

        # temp = self.im_tsdf.copy()
        # temp.V[np.isnan(temp.V)] = np.nanmin(temp.V)


