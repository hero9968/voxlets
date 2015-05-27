'''
combining evaluations
'''

import yaml
import real_data_paths as paths
import numpy as np

modelname = 'v1'

def load_results_for_model(model):
    all_results = []
    for sequence in paths.test_data:
        results_foldername = \
                paths.implicit_predictions_dir % (model, sequence['name'])
        all_results.append(yaml.load(open(results_foldername + 'eval.yaml')))
    return all_results


tests = ['iou', 'precision', 'recall']
models = ['v1', 'autorotate', 'sorted', 'sorted_together',
        'cobweb', 'rays', 'rays_cobweb', 'zheng', 'zheng2']

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
