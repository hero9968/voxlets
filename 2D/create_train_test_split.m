% given a folder of images, splits them between training and test.
% will probably split at the object level so all rotated versions of a 
% shape must be in different spits
% - would also like to split at class level, but am not sure how easy this
% might be

clear
define_params
load(paths.filelist)
test_fraction = params.test_split.test_fraction;
max_test_images = params.test_split.max_test_images;
max_training_images = params.test_split.max_training_images;

%% setting some parameters
number_shapes = length(filelist);
unique_classes = unique({filelist.class});
unique_class_idxs = unique([filelist.class_idx]);

number_classes = length(unique_classes);
number_test_classes = round(test_fraction * number_classes);

%% randomly assigning classes to train and test
perm_idx = randperm(number_classes);
split.test_class_idx = unique_class_idxs(perm_idx(1:number_test_classes));
split.train_class_idx = unique_class_idxs(perm_idx(number_test_classes+1:end));

%% converting the classes to the individual shapes
split.test_idx = find(ismember([filelist.class_idx], split.test_class_idx));
split.train_idx = find(ismember([filelist.class_idx], split.train_class_idx));

%% converting this split to file names

%% creating testing data
split.test_data = [];
for ii = 1:length(split.test_idx)
    this_image_idx = split.test_idx(ii);
    this_filename = sprintf(paths.raytraced_savename, ii);
    split.test_data(ii).file = this_filename;
    split.test_data(ii).image_idx = this_image_idx;
    split.test_data(ii).filelist = filelist(ii);
    
end

if length(split.test_data) > max_test_images
    idx_to_use = randperm(length(split.test_data), max_test_images);
    split.test_data = split.test_data(idx_to_use);
end

%% creating training data
split.train_data = [];
for ii = 1:length(split.train_idx)
    this_image_idx = split.train_idx(ii);
    this_filename = sprintf(paths.raytraced_savename, ii);
    split.train_data(ii).file = this_filename;
    split.train_data(ii).image_idx = this_image_idx;
    split.train_data(ii).filelist = filelist(ii);
end

if length(split.train_data) > max_training_images
    idx_to_use = randperm(length(split.train_data), max_training_images);
    split.train_data = split.train_data(idx_to_use);
end

%% saving this split
save(paths.split_path, 'split')



