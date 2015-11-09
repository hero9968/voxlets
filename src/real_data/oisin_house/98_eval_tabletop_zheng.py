base = "/media/michael/Seagate/phd_projects/volume_completion_data/data/oisin_house/implicit/models/zheng_2/predictions/"

import os, yaml
import numpy as np

all_res = []

for fname in os.listdir(base):
	results = yaml.load(open(base + fname + '/eval.yaml'))
	all_res.append(results)

scores = ['iou', 'precision', 'recall']


def get_mean_score(all_scores, score_type):
    all_this_scores = []
    for sc in all_scores:
        if score_type in sc:
            all_this_scores.append(sc[score_type])

    return np.array(all_this_scores).mean()

for score in scores:

	print "%0.3f" % get_mean_score(all_res, score),
	print " & ",


# now for nyu synthetic

base = "/home/michael/projects/shape_sharing/data/cleaned_3D/renders_yaml_format/silberman_split/implicit/models/zheng_2/predictions/"

all_res = []
for fname in os.listdir(base):
	if os.path.exists(base + fname + '/eval.yaml'):
		results = yaml.load(open(base + fname + '/eval.yaml'))
		all_res.append(results)

print "\n\nNYU synth"
for score in scores:

	print "%0.3f" % get_mean_score(all_res, score),
	print " & ",
