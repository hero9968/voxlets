% this is the pipeline for creating the files to be used by the main
% program.

%% only want to use a subset of all the shapes
create_subset

%% rotate these shapes to multiple angles
rotate_all_masks

%% raytrace these shapes to create depth images
raytrace_all_masks

%% save images to mat file for quciker access
save_images_mat_files

%% creating a train and test split and save to file
create_train_test_split

%% train the structured prediction model
train_model

%% run a specified prediciton algorithm on all files and save results to disk
run_prediction_algorithm

%% evaluate the accuracy of the predictions made
evaluate_predictions
