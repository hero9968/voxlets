

import yaml

A = '\includegraphics[height=\\omaheight, clip=true, trim=20 30 30 5]{'

test_data = yaml.load(open('/media/ssd/data/oisin_house/train_test/test.yaml'))

f = open('table.tex', 'w')

for count, sequence in enumerate(test_data):

    f.write(A + sequence['name'] + '} & \n')

    for view in ['visible', 'gt', 'Medioid', 'pred_remove_excess']:
        imgname = sequence['name'] + '_' + view + '_view_000.png'

        if count==5:
            imgname = imgname.replace('000', '002')

        f.write(A + imgname + '} ')

        if view == 'pred_remove_excess':
            f.write('\\\\\n')
        else:
            f.write('& \n')


f.close()