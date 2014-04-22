% this is the pipeline for creating the files to be used by the main
% program.
% At the start of each script, 'clear' is run ? therefor no variables
% persist in the workspace between scripts run from this pipeline script.

% set up matlabpool here as it only should be done once for the session
cd ~/projects/shape_sharing/2D
matlabpool(4)

%% now using all the shapes. Want to generate list of files
generate_filelist

%% rotate these shapes to multiple angles
rotate_and_raytrace

%% raytrace these shapes to create depth images
% raytracing now occurs in the rotation!
%raytrace_all_masks

%% segemnting the depth images into soup of segments
segment_and_normals

%% creating a train and test split and save to file
create_train_test_split

%% save images to mat file for quciker access
save_images_mat_files

%% train the structured prediction model
train_gaussian_model
train_structured_model

%% run each of the prediction algorithms once to check they run without error
test_prediction_algorithms

%% run a specified prediciton algorithm on all files and save results to disk
run_prediction_algorithm

%% evaluate the accuracy of the predictions made
evaluate_predictions        % save evaluations to disk
view_prediction_outputs     % load evaluations and plot