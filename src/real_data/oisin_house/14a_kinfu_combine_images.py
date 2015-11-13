"""
combine together some kinfu images
"""
import matplotlib.pyplot as plt
import real_data_paths as paths
import scipy.misc
import yaml
import sys

parameters_path = sys.argv[1]
parameters = yaml.load(open(parameters_path))

for sequence in paths.test_data:
    # sequence['name'] = 'saved_00215_[364]' #'saved_00207_[536]'
    # sequence['scene'] = 'saved_00215'
    if sequence['scene'] not in ['saved_00211']:
        continue

    sequence['frames'] = range(0, 150, 10)[::-1]#[0, 25, 50, 75, 100, 125, 150, 175, 200]
    print sequence

    gen_renderpath = paths.kinfu_prediction_img_path % \
            (parameters['batch_name'], sequence['name'], '%s')

    for count, frame_num in enumerate(sequence['frames']):
        count += 10
        # render view of predictoin
        savepaths = []
        savepaths.append(
            gen_renderpath % ('input_%04d_%04d.png' % (count, frame_num)))
        savepaths.append(
            gen_renderpath % ('depth_%04d_%04d.png' % (count, frame_num)))
        savepaths.append(
            gen_renderpath % ('kinfu_%04d_%04d.png' % (count, frame_num)))
        savepaths.append(
            gen_renderpath % ('prediction_%04d_%04d.png' % (count, frame_num)))

        failed = False
        fig = plt.figure()
        for idx, savepath in enumerate(savepaths):

            # check this exists
            im = scipy.misc.imread(savepath)
            plt.subplot(2, 2, idx+1)

            if idx == 0:
                plt.title("%d / %d (ID: %d)" % (
                    count, len(sequence['frames']), frame_num))
            elif idx == 2:
                plt.title('Kinfu')
            elif idx == 3:
                plt.title('Voxlets')

            if idx > 1:
                im = im[100:300, 200:450]

            plt.imshow(im)
            plt.axis('off')

        plt.tight_layout()
        plt.savefig(
            gen_renderpath % ('combined_%04d_%04d.png' % (count, frame_num)))
