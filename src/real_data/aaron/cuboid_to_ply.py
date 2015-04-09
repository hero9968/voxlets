import numpy as np
import struct


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
