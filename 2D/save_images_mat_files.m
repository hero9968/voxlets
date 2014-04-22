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

%% forming training data
N = length(split.train_data);
clear train_data new_train_data
train_data(N * params.n_angles) = ...
    struct('idx', [], 'angle', [], 'raytraced', [], ...
            'segmented', [], 'normals', [], 'image', []);

count = 1;
for ii = 1:N
    
    this_idx = split.train_idx(ii);
    
    this_rotated = all_rotated(this_idx);
    
    M = length(this_rotated.angles);
    
    for jj = 1:M
        
        % extracting each depth image from the structure
        new_train_data.idx = this_idx;
        new_train_data.angle = this_rotated.angles(jj);
        new_train_data.raytraced = this_rotated.raytraced{jj};
        new_train_data.segmented = this_rotated.segmented{jj};
        new_train_data.normals = this_rotated.normals{jj};
        
        temp_image = all_images{this_idx};
        new_train_data.image = ...
            rotate_mask(temp_image, new_train_data.angle, params) > 0;
        
        % adding to the training data structure
        train_data(count) = new_train_data;
        count = count + 1;
    end
    
    done(ii, N, 10)
end

save(paths.train_data, 'train_data');


%% forming training data
N = length(split.test_data);
clear test_data new_test_data train_data new_train_data
test_data(N * params.n_angles) = ...
    struct('idx', [], 'angle', [], 'raytraced', [], ...
            'segmented', [], 'normals', [], 'image', []);
        
count = 1;
for ii = 1:N
    
    this_idx = split.test_idx(ii);
    
    this_rotated = all_rotated(this_idx);
    
    M = length(this_rotated.angles);
    
    for jj = 1:M
        
        % extracting each depth image from the structure
        new_test_data.idx = this_idx;
        new_test_data.angle = this_rotated.angles(jj);
        new_test_data.raytraced = this_rotated.raytraced{jj};
        new_test_data.segmented = this_rotated.segmented{jj};
        new_test_data.normals = this_rotated.normals{jj};
        
        temp_image = all_images{this_idx};
        new_test_data.image = ...
            rotate_mask(temp_image, new_test_data.angle, params) > 0;
        
        % adding to the training data structure
        test_data(count) = new_test_data;
        count = count + 1;
    end
    
    done(ii, N, 10)
end

save(paths.test_data, 'test_data');

