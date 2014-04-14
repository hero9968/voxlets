% a script to save training and test images and depths to mat files...

clear 
define_params
load(paths.split_path, 'split')

%%
clear train_data

N = length(split.train_data);

train_data.depths = cell(1, N);
train_data.images = cell(1, N);
train_data.segments = cell(1, N);

for ii = 1:N

    % loading in the depth for this image
    this_filename = split.train_data{ii};
    this_depth_path = fullfile(paths.raytraced, this_filename);
    this_image_path = fullfile(paths.rotated, this_filename);
    this_segmented_path = fullfile(paths.segmented, [this_filename(1:end-4)]);
    
    temp = load([this_depth_path '.mat']);
    train_data.depths{ii} = double(temp.this_raytraced_depth);
    
    temp = load([this_segmented_path, 'segmented.mat']);
    train_data.segments{ii} = double(temp.segmented);
    
    train_data.images{ii} = imread([this_image_path '.gif']);
    train_data.filename{ii} = this_filename;
    
    done(ii, N)
end

save(paths.train_data, 'train_data');

%%
clear test_data

N = length(split.test_data);

test_data.depths = cell(1, N);
test_data.images = cell(1, N);

for ii = 1:N

    % loading in the depth for this image
    this_filename = split.test_data{ii};
    this_depth_path = fullfile(paths.raytraced, this_filename);
    this_image_path = fullfile(paths.rotated, this_filename);
    this_segmented_path = fullfile(paths.segmented, [this_filename(1:end-4)]);
    
    temp = load([this_depth_path '.mat']);
    test_data.depths{ii} = double(temp.this_raytraced_depth);
    
    temp = load([this_segmented_path, 'segmented.mat']);
    train_data.segments{ii} = double(temp.segmented);
    
    test_data.images{ii} = imread([this_image_path '.gif']);
    test_data.heights(ii) = size(test_data.images{ii}, 1);
    test_data.filename{ii} = this_filename;        

    done(ii, N)
end

save(paths.test_data, 'test_data');

