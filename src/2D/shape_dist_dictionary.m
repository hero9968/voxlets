% Creating a dictionary of `bins` for shape distributions

cd ~/projects/shape_sharing/2D
clear
define_params
addpath(genpath('./src/'))
addpath(genpath('../common/'))

%% loading in all depths and shapes from disk...
load(paths.all_images, 'all_images')
load(paths.train_data, 'train_data')

%% now compute the model
N = length(train_data);
% set up params etc.
num_samples = params.shape_dist.num_samples;

%% loop over each training instance
clear dists angles
train_instances = randperm(N, 1000);
for jj = 1:length(train_instances)

    ii = train_instances(jj);

    this_depth = train_data(ii).depth;
    to_remove = outside_nans(this_depth);
    this_depth(to_remove) = [];
    XY = xy_from_depth(this_depth);
    
    % rescaling the XY points
    xy_bin_edges = params.shape_dist.si_bin_edges;
    train_data(ii).scale = normalise_scale(XY);
    
    XY_scaled = train_data(ii).scale * XY;
    
    % computing the distances and angles ? don't need FV at this stage
    norms = train_data(ii).normals;
    norms(:, to_remove) = [];
    [~, dists{jj}, angles{jj}] = ...
        shape_distribution_2d_angles(XY_scaled, norms, num_samples, xy_bin_edges, params.angle_edges, 1);
end

%% forming the dictionaries
all_dists = cell2mat(dists);
all_angles = cell2mat(angles);

to_use = randperm(length(all_dists), 100000);
all_dists = all_dists(to_use);
all_angles = all_angles(to_use);

%% dictionary with just the distances or just the angles
%[k, dist_dict] = kmeans(to_remove', 100, 'replicates', 10);
%[k, angles_dict] = kmeans(all_angles', 100, 'replicates', 10);

%% disctionary combined over distances and angles
[k, dist_angle_dict] = kmeans([all_dists', all_angles'], 50, 'replicates', 5);%, 'onlineupdate', 'off');

%% saving dictionary
save(paths.dist_angle_dict, 'dist_angle_dict');

%% Plotting the histograms and dictionaries
subplot(131);
hist(all_dists, 100);
title('Distances')

subplot(132);
hist(all_angles, 100);
title('Angles')

subplot(133)
dists_edges = 0:0.05:1.5;
angles_edges = -1:0.05:1;
H = hist2d([all_dists', all_angles'], dists_edges, angles_edges);
imagesc(angles_edges, dists_edges, H)
hold on
plot(dist_angle_dict(:, 2), dist_angle_dict(:, 1), 'r.', 'markersize', 30)
hold off
colormap(gray)
axis image
title('Distances and angles')

%%
[~, idx] = pdist2(dist_angle_dict, [all_dists', all_angles'], 'Euclidean', 'Smallest', 1);
T = accumarray(idx(:), 1);
bar(T)


%% now run the dictionary on some test images
clf
count = 1;
for ii = 1001:1100

    this_depth = train_data(ii).depth;
    to_remove = outside_nans(this_depth);
    this_depth(to_remove) = [];
    XY = xy_from_depth(this_depth);
    
    % rescaling the XY points
    xy_bin_edges = params.shape_dist.si_bin_edges;
    train_data(ii).scale = normalise_scale(XY);
    
    XY_scaled = train_data(ii).scale * XY;
    
    % computing the distances and angles ? don't need FV at this stage
    norms = train_data(ii).normals;
    norms(:, to_remove) = [];

    histogram = ...
        shape_dist_2d_dict(XY_scaled, norms, 10000, dist_angle_dict);

    subplot(10, 10, count);
    count = count + 1;
    bar(histogram)
    set(gca, 'xlim', [0, 50])
end
