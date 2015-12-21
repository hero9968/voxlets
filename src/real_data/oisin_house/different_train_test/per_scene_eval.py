'''
aim is to see which scenes give what scores
'''

import yaml
import numpy as np

import sys, os
sys.path.append(os.path.expanduser("~/projects/shape_sharing/src/real_data/oisin_house/"))
import real_data_paths as paths
seqs = paths.test_data

scores = yaml.load(open('./all_results.yaml'))

all_iou = np.array([score['standard_cobweb']['iou'] for score in scores])

idxs = np.argsort(all_iou)
print scores[idxs[0]]
print all_iou[idxs]

# print np.sort(all_iou)
print np.mean(all_iou)
print np.median(all_iou)
print np.mean(np.sort(all_iou)[130:])
print all_iou.shape

seq_scores = {}
for seq, score in zip(seqs, scores):
    if seq['scene'] in seq_scores:
        seq_scores[seq['scene']].append(score['standard_cobweb']['iou'])
    else:
        seq_scores[seq['scene']] = [score['standard_cobweb']['iou']]

print seq_scores

best_scores = []
for seq, score in seq_scores.iteritems():
    best_scores.append(np.max(np.array(score)))
print np.array(best_scores).mean()
