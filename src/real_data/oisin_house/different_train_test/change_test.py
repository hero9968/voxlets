import yaml
from copy import deepcopy

loadpath = '/media/ssd/data/oisin_house/train_test/test.yaml'
savepath0 = '/media/ssd/data/oisin_house/train_test/test_minus_20.yaml'
savepath1 = '/media/ssd/data/oisin_house/train_test/test_plus_20.yaml'

scenes = yaml.load(open(loadpath))

new0 = []
new1 = []

for scene in scenes:
    dscene = deepcopy(scene)

    dscene['frames'][0] -= 20
    dscene['name'] = dscene['scene'] + '_[' + str(dscene['frames'][0]) + ']'
    new0.append(dscene)

    dscene['frames'][0] += 40
    dscene['name'] = dscene['scene'] + '_[' + str(dscene['frames'][0]) + ']'
    new1.append(dscene)

yaml.dump(new0, open(savepath0, 'w'))
yaml.dump(new1, open(savepath1, 'w'))