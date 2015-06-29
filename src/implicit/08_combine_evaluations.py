'''
combining evaluations
'''

import yaml
import numpy as np

parameters = yaml.load(open('./implicit_params.yaml'))

if parameters['testing_data'] == 'oisin_house':
    import real_data_paths as paths
elif parameters['testing_data'] == 'synthetic':
    import synthetic_paths as paths
else:
    raise Exception('Unknown training data')

modelname = 'v1'

def load_results_for_model(model):
    all_results = []
    for sequence in paths.test_data:
        results_foldername = \
                paths.implicit_predictions_dir % (model, sequence['name'])
        all_results.append(yaml.load(open(results_foldername + 'eval.yaml')))
    return all_results


tests = ['iou', 'precision', 'recall']
# , 'zheng_2', 'zheng_3',
models = [ 'cobweb', 'rays', 'rays_cobweb', 'rays_autorotate', 'zheng_2', 'zheng_3']
#         'rays_cobweb_10',
#         'rays_cobweb_100',
#         'rays_cobweb_1000',
#         'rays_cobweb_5000',
#         'rays_cobweb_10000',
#         'rays_cobweb_20000',
#         'rays_cobweb_50000',
#         'rays_cobweb_100000',
#         'rays_cobweb_500000',
#         'rays_cobweb_1000000']
# 'autorotate', 'sorted_together',


# loading and printing results as a nice table
print ' ' * 20,
for test in tests:
    print test.ljust(10),

print '\n' + '-' * (20+10*3)

for model in models:
    all_results = load_results_for_model(model)
    print model.ljust(20),
    for test in tests:
        avg_result = np.array([result[test] for result in all_results]).mean()
        print ('%0.4f' % avg_result).ljust(10),
    print ''


# do the same in latex
print '\\begin{table}'

print '                     & \\textbf{' + '} & \\textbf{'.join(tests) + '} \\\\'


for model in models:
    all_results = load_results_for_model(model)
    thisstr = model.ljust(20) + ' & '
    for test in tests:
        avg_result = np.array([result[test] for result in all_results]).mean()
        thisstr += ('%0.3f & ' % avg_result)
    print thisstr[:-3] + ' \\\\'

print '\\end{table}'
