% given a folder of images, splits them between training and test.
% will probably split at the object level so all rotated versions of a 
% shape must be in different spits

clear
define_params
load(paths.subset_files)
test_fraction = 0.5;

%% setting some parameters
number_shapes = length(filelist);
image_folder_path = paths.rotated;
test_number = round(test_fraction * number_shapes);

%% randomly assigning shapes to train and test
perm_idx = randperm(number_shapes);
split.test_idx = perm_idx(1:test_number);
split.train_idx = perm_idx(test_number+1:end);

%% converting this split to file names
split.test_data = {};
for ii = split.test_idx
    for jj = 1:params.n_angles
        this_filename = sprintf(paths.rotated_filename, ii, jj);
        split.test_data = [split.test_data, this_filename(1:end-4)];
    end
end

split.train_data = {};
for ii = split.train_idx
    for jj = 1:params.n_angles
        this_filename = sprintf(paths.rotated_filename, ii, jj);
        split.train_data = [split.train_data, this_filename(1:end-4)];
    end
end

%% saving this split
save(paths.split_path, 'split')



