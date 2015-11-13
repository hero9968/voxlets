

import os

A = '\includegraphics[height=\\omaheight, clip=true, trim=170 120 50 30]{'
A2 = '\includegraphics[height=\\omaheight]{'
header = "Input RGB & Input Depth & Observed surfaces & Zheng \\ea (GT) & Bounding box (GT) & \\textbf{Voxlets} & Ground truth  \\\\\n"

rows_per_page = 9

test_data = os.listdir('imgs/tabletop_renders/')

f = open('table.tex', 'w')
# f.write(header)

for count, sequence in enumerate(test_data):

    # f.write(A + test_data + '} & \n')
    if count % rows_per_page == 0:
        if count > 0:
            f.write("\\end{tabular}\n\n")
        f.write("\\begin{tabular}{cccc}\n")
        f.write(header)

    for view in ['input', 'input_depth', 'visible', 'zheng', 'bounding_box', 'short_and_tall_samples_no_segment', 'ground_truth']:

        imgname = sequence + '/' + view + '.png'

        if view in ['input']:
            f.write(A2 + imgname + '} ')
        else:
            f.write(A + imgname + '} ')

        if view == 'ground_truth':
            f.write('\\\\\n')
        else:
            f.write('& \n')

    print count
    # if count > 0 and (count % 9 == 0):# or count -1 % 30 == 0):
    #     f.write(header)

f.write("\\end{tabular}\n")
# f.write(header)
f.close()
