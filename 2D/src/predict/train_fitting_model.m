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
bin_edges = params.shape_dist.bin_edges;

% for each depth, compute a shape distriution feature
shape_dists = cell(1, N);
translations = cell(1, N);
rotations = cell(1, N);
M = cell(1, N);

for ii = 1:N

    Y = (double(depth{ii}));
    X = 1:length(Y);
    
    % computing the shape distributions
    shape_dists{ii} = shape_distribution_2d(X(:), Y(:), num_samples, bin_edges);   
    
    % find the translation and rotation using PCA...
    [translations{ii}, rotations{ii}, M{ii}] = transformation_to_origin_2d(X, Y);
    
end

model.shape_dists = shape_dists;
model.translations = translations;
model.rotations = rotations;
model.transf = M;














