import subprocess as sp

sp.call(['blender', 'data/render_cad.blend',
    '--background', '-P', 'segmented_render_helper.py'])
