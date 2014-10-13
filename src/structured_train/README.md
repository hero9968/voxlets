# Main scripts

The order in which the scripts are run is:

## compute_data.py

This takes as inputs the pairs of depth renderings, and saves to disk the features (e.g. patch and spider features), together with the target depth variables. Features are computed for a random set of points from each image.

## create_train_test_split.py

Title should be self-explaintory

## combine_data.py

Loads all the separate saved files of features and combines into one big file. This allows for much quicker loading when training the forest
There are 4 different types of saved data now: train/test, and small/normal

TODO - consider pickling rather than using .mat, for quicker saving and loading...

## train_model.py

This loads the saved features and target variables, and trains and saves a random forest model. Has a few options for which features to use and how many items to train with. Currently these must be set manually in the script.
Loading and reshaping the data is currently quite slow - a TODO is to split this script into two parts
TODO - think about random selection per-tree
TODO - use the yaml config file to automatically generate models with different parameters

## prediction.ipynb

Doing the actual prediction. Currently as an ipynb so as to allow for the model to be loaded in just once (this is slow!) at the top of the file.
TODO - make this do the prediction for each object in turn
TODO - options for sparse prediction and dense prediction

## sparse_prediction.py

Prediction on the test data, which is extracted sparsely from various test images.
Iterates over each saved model in order to make these predictions

## sparse_evaluation.py

Combines together the various sparse predictions from each of the models, saved into the results files.
Plots ROC curve for each one
Not doing anything else here yet...




# Auxiliary classes etc

## voxel_data.py

Class to store voxel data, and extract slices etc.

## presentation_bits.ipynb

Ipython notebook creating a few auxiliary images for presentations and papers, such as rendering views of objects etc.

## tests/hack_3d_plot.py

Trying out projecting depth images to 3D. Results do not look good so will leave for now