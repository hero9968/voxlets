% this is the pipeline for creating the files to be used by the main
% program.

%% only want to use a subset of all the shapes
create_subset

%% rotate these shapes to multiple angles
rotate_all_masks

%% raytrace these shapes to create depth images
raytrace_all_masks