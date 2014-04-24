% a script to save training and test images and depths to mat files...

clear 
cd ~/projects/shape_sharing/2D
define_params
load(paths.split_path, 'split')
load(paths.filelist, 'filelist')

%% loading in all rotated structures and concatenating

for ii = 1:length(filelist)
    
    this_filename = sprintf(paths.raytraced_savename, ii);
    load(this_filename, 'rotated');
    
    % creating an array of all the structures
    all_rotated(ii) = rotated;
    
    done(ii, length(filelist), 100)
    
end

%% creating all the images togethrer
all_images = {all_rotated.image};
save(paths.all_images, 'all_images')

%% splitting off training data (don't need the segmentation in the training data)
train_data = [all_rotated(split.train_idx).rendered];
train_data = rmfield(train_data, 'segmented');
save(paths.train_data, 'train_data');

%% splitting off test data
test_data = [all_rotated(split.test_idx).rendered];
save(paths.test_data, 'test_data');
