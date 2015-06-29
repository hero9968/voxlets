'''
A collection of functions for rendering voxlets and scenes
'''
import numpy as np
import os, sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage import measure
import cPickle as pickle
import subprocess as sp
import shutil

sys.path.append(os.path.expanduser(
    '~/projects/shape_sharing/src/rendered_scenes/visualisation'))

import voxel_utils as vu

# Setting some paths
if sys.platform == 'darwin':
    blender_path = "/Applications/blender.app/Contents/MacOS/blender"
    font_path = "/Librarry/Fonts/Verdana.ttf"
elif sys.platform == 'linux2':
    blender_path = "blender"
    font_path = "/usr/share/fonts/truetype/msttcorefonts/verdana.ttf"
else:
    raise Exception("Unknown platform...")

voxlet_render_blend = os.path.expanduser(
        '~/projects/shape_sharing/src/rendered_scenes/visualisation/voxlet_render_%s_%s.blend')
voxlet_render_script = os.path.expanduser(
        '~/projects/shape_sharing/src/rendered_scenes/visualisation/single_voxlet_blender_render.py')


def render_single_voxlet(
    V, savepath, level=0, height='tall',
    actually_render=True, speed='quick', keep_obj=False):

    assert V.ndim == 3

    # renders a voxlet using the .blend file...
    temp = V.copy()
    if V.min() > 0 or V.max() < 0:
        print "Level set not present"
        return

    # add a bit around the edge, and adjust origin accordingly...
    crap = ((100.0, 100.0), (100.0, 100.0), (100.0, 100.0))
    temp = np.pad(temp, pad_width=1, mode='constant', constant_values=crap)

    verts, faces = measure.marching_cubes(temp, level)
    # now take off the pad width..
    # verts -= 0.005

    if height == 'tall':
        # verts -= 0.01
        verts -= 1
    else:
        verts -= 1

    if np.any(np.isnan(verts)):
        import pdb; pdb.set_trace()

    D = dict(verts=verts, faces=faces)
    with open('/tmp/vertsfaces.pkl', 'wb') as f:
        pickle.dump(D, f)

    verts *= 0.0175 # parameters.Voxlet.size << bit of a magic number here...
    verts *= 10.0  # so its a reasonable scale for blender
    # print verts.min(axis=0), verts.max(axis=0)
    D = dict(verts=verts, faces=faces)
    with open('/tmp/vertsfaces.pkl', 'wb') as f:
        pickle.dump(D, f)
    vu.write_obj(verts, faces, '/tmp/temp_voxlet.obj')

    blender_filepath = voxlet_render_blend % (speed, height)
    if not os.path.exists(blender_filepath):
        raise Exception("Can't find the blender filepath " + blender_filepath)

    #now copy file from /tmp/.png to the savepath...
    folderpath = os.path.split(savepath)[0]
    if not os.path.exists(folderpath):
        os.makedirs(folderpath)

    if actually_render:
        sp.call([blender_path,
             blender_filepath,
             "-b", "-P",
             voxlet_render_script],
             stdout=open(os.devnull, 'w'),
             close_fds=True)

        print "Moving render to " + savepath
        if not os.path.exists('/tmp/temp_voxlet.png'):
            print "ERROR - can't find file /tmp/temp_voxlet.png"
        else:
            shutil.move('/tmp/temp_voxlet.png', savepath)

    if keep_obj:
        if not os.path.exists('/tmp/temp_voxlet.obj'):
            print "ERROR - can't find file /tmp/temp_voxlet.obj"
        else:
            shutil.move('/tmp/temp_voxlet.obj', savepath + '.obj')


def plot_mesh(verts, faces, ax):
    mesh = Poly3DCollection(verts[faces])
    mesh.set_alpha(0.8)
    mesh.set_edgecolor((1.0, 0.5, 0.5))
    ax.add_collection3d(mesh)

    ax.set_aspect('equal')
    MAX = 20
    for direction in (0, 1):
        for point in np.diag(direction * MAX * np.array([1,1,1])):
            ax.plot([point[0]], [point[1]], [point[2]], 'w')
    ax.axis('off')


def render_leaf_medioids(model, folder_path, max_to_render=1000, tree_id=0, height='tall'):
    '''
    renders all the voxlets at leaf nodes of a tree to a folder
    '''
    leaf_nodes = model.forest.trees[tree_id].compact_leaf_nodes()

    print "There are %d leaf nodes " % len(leaf_nodes)

    print model.training_Y.shape

    if not os.path.exists(folder_path):
        raise Exception("Could not find path %s" % folder_path)


    print "Leaf node shapes are:"
    # Rendering each example
    for idx, node in enumerate(leaf_nodes):
        savepath = folder_path + '/%05d.png' % idx
        print "Inverting transform"
        V = model.pca.inverse_transform(model.training_Y[idx])
        print "Now renering"
        render_single_voxlet(V.reshape(model.voxlet_params['shape']), savepath, height=height)

        if idx > max_to_render:
            break


def render_leaf_nodes(model, folder_path, max_per_leaf=10, tree_id=0):
    '''
    renders all the voxlets at leaf nodes of a tree to a folder
    '''
    leaf_nodes = model.forest.trees[tree_id].leaf_nodes()

    print len(leaf_nodes)

    print "\n Sum of all leaf nodes is:"
    print sum([node.num_exs for node in leaf_nodes])

    print model.training_Y.shape

    if not os.path.exists(folder_path):
        raise Exception("Could not find path %s" % folder_path)

    print "Leaf node shapes are:"
    for node in leaf_nodes:
        print node.node_id, '\t', node.num_exs
        leaf_folder_path = folder_path + '/' + str(node.node_id) + '/'

        if not os.path.exists(leaf_folder_path):
            print "Creating folder %s" % leaf_folder_path
            os.makedirs(leaf_folder_path)

        if len(node.exs_at_node) > max_per_leaf:
            ids_to_render = random.sample(node.exs_at_node, max_per_leaf)
        else:
            ids_to_render = node.exs_at_node

        # Rendering each example at this node
        for count, example_id in enumerate(ids_to_render):
            V = model.pca.inverse_transform(model.training_Y[example_id])
            render_single_voxlet(V.reshape(model.voxlet_params['shape']),
                leaf_folder_path + str(count) + '.png')

        with open(leaf_folder_path + ('total_%d.txt' % len(node.exs_at_node)), 'w') as f:
            f.write(str(len(node.exs_at_node)))

        # # Now doing the average mask and plotting slices through it
        # mean_mask = self._get_mean_mask(ids_to_render)
        # plt.figure(figsize=(10, 10))
        # for count, slice_id in enumerate(range(0, mean_mask.shape[2], 10)):
        #     if count+1 > 3*3: break
        #     plt.subplot(3, 3, count+1)
        #     plt.imshow(mean_mask[:, :, slice_id], interpolation='nearest', cmap=plt.cm.gray)
        #     plt.clim(0, 1)
        #     plt.title('Slice_id = %d' % slice_id)
        #     plt.savefig(leaf_folder_path + 'slices.pdf')
        #     plt.close()


def plot_slices(voxlet, mask, savepath):

    for axis in [0, 1, 2]:

        slice_id = voxlet.shape[axis] / 2

        # Plotting voxlet
        plt.subplot(3, 2, axis*2 + 1)
        plt.imshow(
            voxlet.take(slice_id, axis=axis),
            interpolation='nearest', cmap=plt.cm.bwr)

        maxi = np.max(np.abs(voxlet))
        plt.clim(-maxi, maxi)
        plt.title('Voxlet axis %d' % axis)

        # Plotting mask
        plt.subplot(3, 2, axis*2 + 2)
        plt.imshow(
            mask.take(slice_id, axis=axis),
            interpolation='nearest', cmap=plt.cm.bwr)

        plt.clim(0, 1)
        plt.title('Mask axis %d' % axis)

    plt.savefig(savepath)
    plt.close()

    # mean_mask = self._get_mean_mask(ids_to_render)
        # plt.figure(figsize=(10, 10))
        # for count, slice_id in enumerate(range(0, mean_mask.shape[2], 10)):
        #     if count+1 > 3*3: break
        #     plt.subplot(3, 3, count+1)
        #     plt.imshow(mean_mask[:, :, slice_id], interpolation='nearest', cmap=plt.cm.gray)
        #     plt.clim(0, 1)
        #     plt.title('Slice_id = %d' % slice_id)
        #     plt.savefig(leaf_folder_path + 'slices.pdf')
        #     plt.close()