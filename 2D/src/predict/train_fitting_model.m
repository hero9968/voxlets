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
    norms = training_data(ii).normals;

%{
    to_remove = isnan(training_data(ii).depth);
    XY(:, to_remove) = [];
    norms(:, to_remove) = [];
    
    % rescaling the XY points
    if params.scale_invariant
        training_data(ii).scale = normalise_scale(XY);
    else
        training_data(ii).scale = 1;
    end
    %}
    %XY_scaled = training_data(ii).scale * XY;
    
    % computing the shape distributions
    %training_data(ii).shape_dist = shape_dist_2d_dict(XY_scaled, norms, num_samples, params.dist_angle_dict);

    % find the translation and rotation using PCA...
    [~, ~, training_data(ii).transform_to_origin] = transformation_to_origin_2d(XY);
    
end

% the model consists of the training data with the feature vectors and the images
model.images = training_images;
model.training_data = training_data;

% adding the parameters etc to the model
model.scale_invariant = params.scale_invariant;
model.dist_angle_dict = params.dist_angle_dict;
