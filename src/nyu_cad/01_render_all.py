import subprocess as sp

sp.call(['blender', 'data/render_cad.blend',
    '--background', '-P', 'render_helper.py'])
