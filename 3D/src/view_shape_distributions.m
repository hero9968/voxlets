% script to compute some shape distributions and visualise them...

clear
cd ~/projects/shape_sharing/3D/src/
addpath plotting/
addpath features/
addpath ./file_io/matpcl/
addpath ./file_io
addpath ../../common/
addpath transformations/
addpath ../../2D/src/segment/
run ../define_params_3d.m

%% loading in some of the ECCV dataset
filepath = '/Users/Michael/data/others_data/ECCV_dataset/pcd_files/frame_20111220T111153.549117.pcd';
cloud = loadpcd_as_cloud(filepath);

%% computing normals
[cloud.normals, cloud.curvature] = normals_wrapper(cloud.xyz, 'knn', 50);

%% running segment soup algorithm
[idxs, idxs_without_nans] = segment_soup_3d(cloud, params.segment_soup);

%% plot shape distribution for each segment
close all
plot_segment_soup_3d(cloud.rgb, idxs);

%% compute and plot the shape distributions
sd_opts.bin_edges = linspace(0, 5, 50);
sd_opts.num_samples = 5000;
sd_opts.rescaling = true;

[n, m] = best_subplot_dims(size(idxs, 2));

shape_dist = cell(1, size(idxs, 2));

for ii = 1:size(idxs, 2)
    
    this_xyz = cloud.xyz(idxs(:, ii)==1, :);
    shape_dist{ii} = shape_distribution_3d(this_xyz, sd_opts);
    
    %rescaled_bins = rescale_histogram(this_shape_dist', shape_dist_bin_edges);
    
    subplot(n, m, ii);
    bar(sd_opts.bin_edges, shape_dist{ii})
    set(gca, 'xlim', [min(shape_dist_bin_edges), max(shape_dist_bin_edges)])
    
end

%% plot the distances between different shape distributions
figure
all_hists = cell2mat(shape_dist)';
T = squareform(pdist(all_hists, 'cityblock'));
imagesc(T)
colormap((gray))
axis image
colorbar

