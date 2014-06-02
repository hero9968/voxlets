% Computing all the feature vectors. The aim is to end up with them as
% a single .mat file for each of the 1600 or so objects...

cd ~/projects/shape_sharing/3D/
clear
addpath(genpath('.'))
run define_params_3d.m

render_path = '/Users/Michael/projects/shape_sharing/data/3D/basis_models/renders/';
fv_path = '/Users/Michael/projects/shape_sharing/data/3D/basis_models/fv/';
models_to_use = randperm(length(params.model_filelist), 100);
renders_to_use = randperm(42, 10);

%%
pairwise_dists = cell(1, length(renders_to_use));

for ii = 1:length(models_to_use)

    % getting the path to the renders
    model = params.model_filelist{models_to_use(ii)};
    render_dir = fullfile(render_path, model);
    depth_names = fullfile(render_dir, 'depth_%d.mat');
        
    % loop over each image and combine all the results together
    for jj = 1:length(renders_to_use)
        
        % load depth and project to 3D
        this_idx = renders_to_use(jj);
        this_name = sprintf(depth_names,this_idx);
        load(this_name, 'depth');
        this_xyz = reproject_depth(depth, params.half_intrinsics, 3);
        this_xyz = this_xyz * normalise_scale(this_xyz);        

        % compute the pairwise distances
        params.shape_dist.rescaling = 0;
        [~, pairwise_dists{ii, jj}] = shape_distribution_3d(this_xyz, params.shape_dist);
    end
    
    disp(['Done ' num2str(ii)]);
end

%% Seeing how many samples seemed sensible to take to normalise scale
%profile on
num_samples = round(10.^[linspace(1, 6, 50)]);
clear t
for ii = 1:length(num_samples)
    tic
    t(ii) = normalise_scale(this_xyz, num_samples(ii));
    times(ii) = toc;
end
semilogx(num_samples, t);
hold on
plot(num_samples, times)

%profile off viewer

%% forming the dictionary
all_dists = cell2mat(pairwise_dists(:));
all_dists_subset = all_dists(randperm(length(all_dists), 20000));
[~, dict] = kmeans(all_dists_subset, 50, 'replicates', 20);
dict = sort(dict);

%% plotting dictionary
hist(all_dists, 100);
hold on
plot(dict, 0*dict, 'r+')
hold off

%% saving dictionary
save(paths.shape_dist_dict, 'dict');





