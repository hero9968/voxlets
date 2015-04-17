from sklearn.cross_validation import train_test_split
import real_data_paths as paths
import yaml

test_size = 0.25

test_N = int(test_size * float(len(paths.scenes)))
train_N = len(paths.scenes)

train, test = train_test_split(paths.sequences, test_size=test_size, random_state=10)

import itertools
train = [t for temp in train for t in temp]
test = list(itertools.chain(*test))

print len(train)
print len(test)

with open(paths.yaml_train_location, 'w') as f:
    yaml.dump(train, f)

with open(paths.yaml_test_location, 'w') as f:
    yaml.dump(test, f)

