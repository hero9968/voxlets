import subprocess as sp
import parameters
import paths
from time import time
import os

def render_seq(i):
    sp.call([
        paths.blender_path,
        "data_generation/data/blank.blend",
        "-b",
        "-P",
        "data_generation/physics.py"],
        stdout=open(os.devnull, 'w'),
        close_fds=True)
    print "Done ", i


if parameters.multicore:
    # need to import these *after* pool_helper has been defined
    import multiprocessing
    pool = multiprocessing.Pool(parameters.cores)
    mapper = pool.map
else:
    mapper = map

if __name__ == "__main__":

    tic = time()
    mapper(render_seq, range(parameters.RenderData.scenes_to_render))
    print "In total took %f s" % (time() - tic)
