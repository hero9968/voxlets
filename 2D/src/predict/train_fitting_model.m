function model = train_fitting_model(training_images, depth, params)
% given a cell array of training images and a cell array of depths rendered
% from the images, this function trains a model

% input checks
assert(iscell(training_images))
assert(iscell(depth))
assert(length(training_images)==length(depth));
N = length(training_images);

% set up params etc.
num_samples = params.shape_dist.num_samples;

% for each depth, compute a shape distriution feature
shape_dists = cell(1, N);
translations = cell(1, N);
rotations = cell(1, N);
M = cell(1, N);
scale = nan(1, N);

for ii = 1:N

    Y = (double(depth{ii}));
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
    tY = scale(ii) * X(:);
    shape_dists{ii} = shape_distribution_2d(tX, tY, num_samples, bin_edges);
    
    % find the translation and rotation using PCA...
    [translations{ii}, rotations{ii}, M{ii}] = transformation_to_origin_2d(X, Y);
    
end

model.shape_dists = shape_dists;
model.translations = translations;
model.rotations = rotations;
model.transf = M;
model.scale_invariant = params.scale_invariant;
model.bin_edges = bin_edges;
model.scales = scale;













