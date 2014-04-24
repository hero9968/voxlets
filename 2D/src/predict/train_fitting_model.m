function model = train_fitting_model(training_images, training_data, params)
% given a cell array of training images and a cell array of depths rendered
% from the images, this function trains a model

% input checks
assert(iscell(training_images))
N = length(training_data);

% set up params etc.
num_samples = params.shape_dist.num_samples;

% for each depth, compute a shape distriution feature
shape_dists = cell(1, N);
translations = cell(1, N);
rotations = cell(1, N);
M = cell(1, N);
scale = nan(1, N);

for ii = 1:N

    Y = training_data(ii).depth;
    X = 1:length(Y);
    
    % computing the shape distributions
    if params.scale_invariant
        bin_edges = params.shape_dist.si_bin_edges;
        scale(ii) = normalise_scale([X;Y]);
    else
        bin_edges = params.shape_dist.bin_edges;
        scale(ii) = 1;
    end
    
    tX = scale(ii) * X(:);
    tY = scale(ii) * Y(:);
    
    if params.sd_angles
        %fv = shape_distribution_2d_angles(XY, norms, num_samples, xy_bin_edges, angles_bin_edges)
        %norms = normals_radius_2d([tX'; tY'], scale(ii) * params.normal_radius);
        norms = training_data(ii).normals;
        xy_bin_edges = bin_edges;
        angles_bin_edges = params.angle_edges;
        shape_dists{ii} = shape_distribution_2d_angles([tX'; tY'], norms, num_samples, xy_bin_edges, angles_bin_edges);
    else
        shape_dists{ii} = shape_distribution_2d(tX, tY, num_samples, bin_edges);
    end
    
    % find the translation and rotation using PCA...
    [translations{ii}, rotations{ii}, M{ii}] = transformation_to_origin_2d(X, Y);
    
end

model.shape_dists = shape_dists;
model.translations = translations;
model.rotations = rotations;
model.transf = M;
model.scale_invariant = params.scale_invariant;
model.bin_edges = bin_edges;
model.angle_edges = params.angle_edges;
model.scales = scale;
model.sd_angles = params.sd_angles;

model.images = training_images;
%model.depths = {training_data.depth};
model.training_data = training_data;
%model.image_idxs = {training_data.image_idx};
%model.image_transforms = {training_data.transform};












