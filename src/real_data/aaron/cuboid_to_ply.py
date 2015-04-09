import numpy as np
import struct
import os
import matplotlib.pyplot as plt


def read_cuboids(fpath):

    with open(fpath, 'r') as f:
        # first number is a 32 bit int
        number_cuboids = struct.unpack('i', f.read(4))[0]

        # each cuboid is 8*3 32bit floats
        return [np.fromfile(f, dtype=np.float32, count=8*3).reshape((-1, 3))
                for i in range(number_cuboids)]


def read_depth(fpath):

    with open(fpath) as f:
        f.read(8)  # throw away some header
        return np.fromfile(f, dtype=np.float32).reshape(480, 640)


def write_obj(verts, faces, file_name):
    # write obj to file
    f = open(file_name, 'w')
    for ii in range(verts.shape[0]):
        f.write('v ' + str(verts[ii, 0]) + ' ' + str(verts[ii, 1]) + ' ' + str(verts[ii, 2]) + '\n')
    for ii in range(faces.shape[0]):
        f.write('f ' + str(faces[ii, 0]) + ' ' + str(faces[ii, 1]) + ' ' + str(faces[ii, 2]) + ' ' + str(faces[ii, 3]) + '\n')
    f.close()


def create_obj(data_dir, file_name):
    # hand defined face triangulation
    faces = np.asarray(([1, 5, 8, 4], [5, 6, 7, 8], [6, 2, 3, 7], [2, 1, 4, 3], [3, 4, 8, 7], [1, 2, 6, 5])).astype('int')

    if os.path.isfile(data_dir + file_name):
        coords = read_cuboids(data_dir + file_name)

        verts_full = np.concatenate(coords)
        faces_full = faces.copy()
        for ff in range(1, len(coords)):
            faces_full = np.vstack((faces_full, faces + ff*8))

        write_obj(verts_full, faces_full, data_dir + file_name[:-4] + '.obj')


if __name__ == '__main__':
    # TODO different cubes seem to have different face orientation

    # running this will create an obj for both optimized and cuboids.dat and save it to the same
    # directory as the input
    data_dir = '/home/omacaodh/Downloads/imaginingTheUnseen_dataRecorded/'  # points to the root of the scenes
    dirs = next(os.walk(data_dir))[1]

    for scene_name in dirs:
        create_obj(data_dir + scene_name + '/', 'cuboids.dat')
        create_obj(data_dir + scene_name + '/', 'optimized.dat')

