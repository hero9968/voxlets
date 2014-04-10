% given a folder of images, splits them between training and test.
% will probably split at the object level so all rotated versions of a 
% shape must be in different spits
% - would also like to split at class level, but am not sure how easy this
% might be

clear
define_params
load(paths.subset_files)
test_fraction = 0.1;
max_test_images = 500;
max_training_images = 2000;

%% setting some parameters
number_shapes = length(filelist);
unique_classes = unique({filelist.class});
unique_class_idxs = unique([filelist.class_idx]);

number_classes = length(unique_classes);
number_test_classes = round(test_fraction * number_shapes);

%% randomly assigning classes to train and test
perm_idx = randperm(number_classes);
split.test_class_idx = unique_class_idxs(perm_idx(1:number_test_classes));
split.train_class_idx = unique_class_idxs(perm_idx(number_test_classes+1:end));

%% converting the classes to the individual shapes
split.test_idx = find(ismember([filelist.class_idx], split.test_class_idx));
split.train_idx = find(ismember([filelist.class_idx], split.train_class_idx));

%% converting this split to file names
split.test_data = {};
for ii = split.test_idx
    for jj = 1:params.n_angles
        this_filename = sprintf(paths.rotated_filename, ii, jj);
        split.test_data = [split.test_data, this_filename(1:end-4)];
    end
end

if length(split.test_data) > max_test_images
    idx_to_use = randperm(length(split.test_data), max_test_images);
    split.test_data = split.test_data(idx_to_use);
end


split.train_data = {};
for ii = split.train_idx
    for jj = 1:params.n_angles
        this_filename = sprintf(paths.rotated_filename, ii, jj);
        split.train_data = [split.train_data, this_filename(1:end-4)];
    end
end

if length(split.train_data) > max_training_images
    idx_to_use = randperm(length(split.train_data), max_training_images);
    split.train_data = split.train_data(idx_to_use);
end

%% saving this split
save(paths.split_path, 'split')



