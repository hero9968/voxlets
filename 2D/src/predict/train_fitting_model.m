function model = train_fitting_model(training_images, training_data, params)
% given a cell array of training images and a cell array of depths rendered
% from the images, this function trains a model

% input checks
assert(iscell(training_images))
N = length(training_data);

% set up params etc.
num_samples = params.shape_dist.num_samples;

% loop over each training instance
for ii = 1:N

    XY = xy_from_depth(training_data(ii).depth);
    
    % rescaling the XY points
    if params.scale_invariant
        xy_bin_edges = params.shape_dist.si_bin_edges;
        training_data(ii).scale = normalise_scale(XY);
    else
        xy_bin_edges = params.shape_dist.bin_edges;
        training_data(ii).scale = 1;
    end
    
    XY_scaled = training_data(ii).scale * XY;
    
    % computing the shape distributions
    if params.sd_angles == 1
        norms = training_data(ii).normals;
        training_data(ii).shape_dist = ...
            shape_distribution_2d_angles(XY_scaled, norms, num_samples, xy_bin_edges, params.angle_edges, 1);
        
    elseif params.sd_angles == 0
        training_data(ii).shape_dist = shape_distribution_2d(XY_scaled, num_samples, xy_bin_edges);
        
    elseif params.sd_angles == 2
        norms = training_data(ii).normals;
        training_data(ii).shape_dist = ...
            shape_distribution_2d_angles(XY_scaled, norms, num_samples, xy_bin_edges, params.angle_edges, 0);
        
    end
    
    % find the translation and rotation using PCA...
    [~, ~, training_data(ii).transform_to_origin] = transformation_to_origin_2d(XY);
    
end

% the model consists of the training data with the feature vectors and the images
model.images = training_images;
model.training_data = training_data;

% adding the parameters etc to the model
model.scale_invariant = params.scale_invariant;
model.angle_edges = params.angle_edges;
model.sd_angles = params.sd_angles;
model.xy_bin_edges = xy_bin_edges;
