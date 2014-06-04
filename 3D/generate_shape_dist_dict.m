% Computing all the feature vectors. The aim is to end up with them as
% a single .mat file for each of the 1600 or so objects...

cd ~/projects/shape_sharing/3D/
clear
addpath(genpath('.'))
run define_params_3d.m

models_to_use = randperm(length(params.model_filelist), 100);
renders_to_use = randperm(42, 10);

opts = params.shape_dist;
opts.just_dists = 1;

%%
dists = cell(length(models_to_use), length(renders_to_use));
angles = cell(length(models_to_use), length(renders_to_use));
edge_dists = cell(length(models_to_use), length(renders_to_use));
max_depth = nan(length(models_to_use), length(renders_to_use));
%%
for ii = 1:length(models_to_use)

    % getting the path to the renders
    model = params.model_filelist{models_to_use(ii)};
        
    % loop over each image and combine all the results together
    for jj = 1:length(renders_to_use)
        
        % load depth and project to 3D
        this_idx = renders_to_use(jj);
        this_name = sprintf(paths.basis_models.rendered, model, this_idx);
        this_norms_name = sprintf(paths.basis_models.normals, model, this_idx);
        
        load(this_name, 'depth');
        load(this_norms_name, 'normals');
        
        max_depth(ii, jj) = max(depth(:));
        [this_xyz, mask] = reproject_depth(depth, params.half_intrinsics, max_depth(ii, jj));
        
        % this is a hack because sometimes the normals are wrong...
        if abs(max_depth(ii, jj)-2)<0.01
            normals_to_use = depth~=max_depth(ii, jj);
            normals = normals(normals_to_use(:), :);
        end
        
        if ~isempty(this_xyz)
            this_xyz = this_xyz * normalise_scale(this_xyz);
                    
            % compute the pairwise distances
            [~, dists{ii, jj}, angles{ii, jj}] = shape_distribution_norms_3d(this_xyz, normals, opts);           
            [~, edge_dists{ii, jj}] = edge_shape_dists(mask);
            
        end
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

%% forming the 3D dictionary
to_use = ~cellfun(@isempty, dists) & ~cellfun(@isempty, angles);
dist_angles = [cell2mat(dists(to_use(:))), cell2mat(angles(to_use(:)))];
to_use = randperm(length(dist_angles), 20000);
dist_angle_subset = dist_angles(to_use, :);
[idxs, dict] = kmeans(dist_angle_subset, 75, 'replicates', 20);
sizes = (accumarray(idxs, 1));
dict = sortrows(dict);

%% forming the edge dictionary
to_use = ~cellfun(@isempty, edge_dists);
all_edge_dists = cell2mat(edge_dists(to_use(:)));
to_use = randperm(length(all_edge_dists), 20000);
all_edge_dists_subset = all_edge_dists(to_use);
[idxs, edge_dict] = kmeans(all_edge_dists_subset, 75, 'replicates', 20);
edge_dict = sort(edge_dict);

%% plotting edge dictionary
hist(all_edge_dists_subset, 100)
hold on
plot(edge_dict, 0*edge_dict, 'r+')
hold off

%% plotting 3D dictionary
dists_edges = 0:0.05:1.5;
angles_edges = -1:0.025:1;
H = hist2d(dist_angles, dists_edges, angles_edges);
imagesc(angles_edges, dists_edges, H)
hold on
for ii = 1:length(dict)
    plot(dict(ii, 2), dict(ii, 1), 'ro', 'markersize', sizes(ii)/40 + 5)
end
hold off
colormap(gray)
axis image
title('Distances and angles')

%% saving dictionary
save(paths.shape_dist_dict, 'dict', 'edge_dict');

