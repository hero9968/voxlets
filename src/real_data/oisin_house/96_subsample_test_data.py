# here replace the test.yaml with one with just one image from each scene or something....


import yaml
import real_data_paths as paths
with open(paths.yaml_test_location + '.full') as f:
    D = yaml.load(f)

unique_scenes = list(set([d['scene'] for d in D]))
print unique_scenes

seq_to_scene = {}

new_seqs = {}

num_to_choose = 1
for d in D:
    if d['scene'] in seq_to_scene and seq_to_scene[d['scene']] >= num_to_choose:
        # already in...
        if new_seqs[d['scene']]['frames'][0] < d['frames'][0]:
            # replace...
            new_seqs[d['scene']] = d
        else:
            continue
    else:
        new_seqs[d['scene']] = d
        if d['scene'] in seq_to_scene:
            seq_to_scene[d['scene']] += 1
        else:
            seq_to_scene[d['scene']] = 1

new_seqs = [new_seqs[tt] for tt in new_seqs]
print "\n\n\n"
print len(new_seqs)
print len(D)
print len(unique_scenes)

with open(paths.yaml_test_location, 'w') as f:
    yaml.dump(new_seqs, f)