from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage import measure
import matplotlib.pyplot as plt
import numpy as np


def write_obj(verts, faces, filename):

    # probable a faster way of doing this
    obj_file = open(filename, "w")

    for nn in range(verts.shape[0]):
    #   obj_file.write('vn ' + str(norm[nn, 0]) + ' ' + str(norm[nn, 1]) + ' ' + str(norm[nn, 2]) + '\n')
        obj_file.write('v ' + str(verts[nn, 0]) + ' ' + str(verts[nn, 1]) + ' ' + str(verts[nn, 2]) + '\n')

    obj_file.write('\n')
    faces_p = faces + 1
    for ff in range(faces_p.shape[0]):
        obj_file.write('f ' + str(faces_p[ff, 0]) + ' ' + str(faces_p[ff, 1]) + ' ' + str(faces_p[ff, 2]) + '\n')
        #obj_file.write('f ' + str(faces_p[ff, 0]) + '//' + str(faces_p[ff, 0]) + ' ' + str(faces_p[ff, 1]) + '//' + str(faces_p[ff, 1]) + ' ' + str(faces_p[ff, 2]) + '//' + str(faces_p[ff, 2]) + '\n')
    obj_file.close()


def display_mesh_of_vol(vol, countour_level, title_text, figure_id):

    # create mesh and display
    verts, faces = measure.marching_cubes(vol, countour_level)
    fig = plt.figure(figure_id, figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    mesh = Poly3DCollection(verts[faces])
    ax.add_collection3d(mesh)

    ax.set_xlabel("x-axis")
    ax.set_ylabel("y-axis")
    ax.set_zlabel("z-axis")

    ax.set_xlim(0, vol.shape[0])
    ax.set_ylim(0, vol.shape[1])
    ax.set_zlim(0, vol.shape[2])

    plt.title(title_text)
    plt.show()

    return verts, faces


def remove_duplicates(array_data, return_index=False, return_inverse=False):

    """Removes duplicate rows of a multi-dimensional array. Returns the
    array with the duplicates removed. If return_index is True, also
    returns the indices of array_data that result in the unique array.
    If return_inverse is True, also returns the indices of the unique
    array that can be used to reconstruct array_data.
    From https://gist.github.com/jterrace/1337531
    """
    unique_array_data, index_map, inverse_map = np.unique(
            array_data.view([('', array_data.dtype)] * \
                    array_data.shape[1]), return_index=True,
                    return_inverse=True)

    unique_array_data = unique_array_data.view(
            array_data.dtype).reshape(-1, array_data.shape[1])

    # unique returns as int64, so cast back
    index_map = np.cast['uint32'](index_map)
    inverse_map = np.cast['uint32'](inverse_map)

    if return_index and return_inverse:
        return unique_array_data, index_map, inverse_map
    elif return_index:
        return unique_array_data, index_map
    elif return_inverse:
         return unique_array_data, inverse_map


def create_voxel_grid(vol, thresh):
    """
    Given a 3D numpy array vol containing 1s or 0s. This will return the vertices and
    faces of the triangular mesh with a cuboid in the location of a 1.

    TODO:
    remove duplicate faces
    allow for the creation of different size cuboids
    """

    # uses to permute vertices
    # TODO fix this so that it can be bigger than 1
    size_const = 0.5
    arr = [[0, 0, 0],
           [1, 0, 0],
           [0, 1, 0],
           [1, 1, 0],
           [0, 0, 1],
           [1, 0, 1],
           [0, 1, 1],
           [1, 1, 1]]
    arr = np.asarray(arr) - size_const

    # set up faces
    cube_faces = [[2, 1, 0],
                 [2, 3, 1],
                 [7, 3, 2],
                 [6, 7, 2],
                 [3, 7, 1],
                 [7, 5, 1],
                 [4, 7, 6],
                 [4, 5, 7],
                 [1, 4, 0],
                 [5, 4, 1],
                 [4, 6, 2],
                 [4, 2, 0]]
    cube_faces = np.asarray(cube_faces)

    verts = []
    faces = []
    x, y, z = np.where(vol >= thresh)
    for vv in range(len(x)):
        pt = np.tile(np.hstack((x[vv], y[vv], z[vv])), (8, 1))
        verts.append(pt + arr)
        faces.append(cube_faces + vv*8)

    verts = np.vstack(verts) + size_const  # just to make sure that the min vert is 0
    faces = np.vstack(faces)

    #TODO remove duplicate faces
    #verts_unique, index_map, inverse_map = remove_duplicates(verts, True, True)

    return verts, faces


def read_kinect_pgm(filename):

    in_file = open(filename)

    header = None
    size = None
    max_gray = None
    data = []

    for line in in_file:
        stripped = line.strip()

        if stripped[0] == '#':
            continue
        elif header == None:
            if stripped != 'P2':
                return None
            header = stripped
        elif size == None:
            size = map(int, stripped.split())
        elif max_gray == None:
            max_gray = int(stripped)
        else:
            for item in stripped.split():
                data.append(int(item.strip()))

    data = np.reshape(data, (size[1], size[0]))
    return data