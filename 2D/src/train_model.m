% a script to train a model from the training data...

clear
cd ~/projects/shape_sharing/2D/src
run('../define_params')
load(paths.split_path, 'split')
addpath predict utils
cd ~/projects/shape_sharing/2D/src

%% loading in all depths and shapes from disk...
N = length(split.train_data);

depths = cell(1, N);
images = cell(1, N);

for ii = 1:N

    % loading in the depth for this image
    this_filename = split.train_data{ii};
    this_depth_path = fullfile(paths.raytraced, this_filename);
    this_image_path = fullfile(paths.rotated, this_filename);
    
    depths{ii} = imread([this_depth_path '.png']);
    depths{ii} = smooth(double(depths{ii}));
    images{ii} = imread([this_image_path '.gif']);
    
    ii

end

%% now compute the model
run ../define_params
model = train_fitting_model(images, depths, params);
all_dists = cell2mat(model.shape_dists);

imagesc(all_dists)