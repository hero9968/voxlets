from sklearn.cross_validation import train_test_split
import real_data_paths as paths
import yaml

test_size = 0.25

train, test = train_test_split(paths.scenes, test_size=test_size, random_state=10)

with open(paths.yaml_train_location, 'w') as f:
    yaml.dump(train, f)

with open(paths.yaml_test_location, 'w') as f:
    yaml.dump(test, f)

