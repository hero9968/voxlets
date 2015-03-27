# Script to copy specified images to a different folder and create a
# latex file to display them nicely...


import sys
import os
sys.path.append(os.path.expanduser("~/projects/shape_sharing/src/"))
import shutil

from common import paths

to_use = ['umk4nke6pzebef2b_SEQ', 'xf2hcoes8lp9fb1t_SEQ', 'kmrkmma8u2456lgk_SEQ']
test_type = 'oma_implicit'
savefolder = paths.RenderedData.rendered_arrangements_path + 'latex/'
latexfile = savefolder + 'results_table.tex'


if not os.path.exists(savefolder):
    os.makedirs(savefolder)

# This will store the latex
latex = ['\\documentclass[12pt]{article}',
         '\\usepackage{graphicx}',
         '\\begin{document}',
         '\\newcommand{\\turnheight}{0.23\columnwidth}',
         '\\begin{figure*}',
         '\\begin{tabular}{cccc}']

for seq_name in to_use:

    # Get the sequence with this name
    sequence = [s for s in paths.RenderedData.test_sequence() if s['name'] == seq_name][0]

    gen_renderpath = paths.RenderedData.voxlet_prediction_img_path % \
            (test_type, sequence['name'], '%s')

    # Copying the images
    for imagetype in ['input', 'visible', 'gt', 'pred_voxlets']:

        in_path = gen_renderpath % imagetype
        out_fname = seq_name + '_' + imagetype + '.png'
        out_path = savefolder + out_fname
        shutil.copy(in_path, out_path)

        latex.append('\\includegraphics[height=\\turnheight]{%s} &' % out_fname)

    latex[-1] = latex[-1].replace('&', '\\\\')

latex.append('\\footnotesize Input view of scene &')
latex.append('\\footnotesize Input data in 3D space &')
latex.append('\\footnotesize Ground truth occupancy &')
latex.append('\\footnotesize Our reconstruction \\')

latex.append('\\end{tabular}')
latex.append('\\end{figure*}')
latex.append('\\end{document}')

with open(latexfile, 'w') as f:
    for l in latex:
        f.write(l + '\n')
