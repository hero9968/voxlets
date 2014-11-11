% This is a new script to load in RGB, depth and mask from bigbird
% Then reproject RGB into depth
% Then smooth
% Then crop
% Then save
addpath('~/projects/shape_sharing/src/matlab/matlab_features')
addpath('~/projects/shape_sharing/src/preprocess/bigbird/crop_smooth/')
addpath('~/projects/shape_sharing/src/preprocess/edges_and_compass/')
OVERWRITE = false;
%matlabpool(6)
%%
base_path = get_base_path();
[views, modelnames] = get_views_models();

for model_idx = 1:length(modelnames)
    
    modelname = modelnames{model_idx};
    
    % creating folder if need be
    model_folder = [base_path, 'bigbird_cropped/', modelname, '/'];
    if ~exist(model_folder, 'dir')
        disp(['Creating ' model_folder])
        mkdir(model_folder)
    end
    
    for view_idx = 1:length(views)
        
        view = views{view_idx};
        save_name = [model_folder, view, '.mat'];
        
        % seeing if file already exists
        if ~OVERWRITE && exist(save_name, 'file')
            disp(['Skipping ' save_name])
            continue
        end
        
        % loading, cropping
        try
            bb = load_bigbird(modelname, view);
            bb_cropped = reproject_crop_and_smooth(bb);

            bb_cropped.norms = normals_wrapper(bb_cropped.xyz, 'knn', 100);

            flying_norms = reshape(bb_cropped.norms(:, 3), size(bb_cropped.grey)) > -0.4; 
            se = strel('disk',1);
            bb_cropped.mask = imerode(bb_cropped.mask & ~flying_norms, se);

            bb_cropped.edges = imdilate(edge(bb_cropped.mask), se);

            bb_cropped.spider = spider_wrapper(bb_cropped.xyz, bb_cropped.norms, ...
                                                bb_cropped.edges, bb_cropped.mask, bb_cropped.T.K_rgb(1));

        catch
            disp(['Failed ', modelname, view])
        end
        % saving to disk
        save_file(save_name, bb_cropped);

        fprintf('%d ', view_idx);
      
    end
    
    disp(['Done model ', modelname, ' ', num2str(model_idx)])
    
end

%plotNormals(bb_cropped.xyz, bb_cropped.normals, 0.01)


%% Displaying the elements of one of the things
inliers = bb_cropped.xyz(bb_cropped.mask(:), :);

subplot(231)
imagesc(bb_cropped.rgb)
axis image
subplot(232)
imagesc(bb_cropped.depth)
axis image
subplot(233)
imagesc(bb_cropped.mask)
axis image
subplot(234)
imagesc(bb_cropped.edges)
%plot(inliers(:, 1), inliers(:, 2), 'o')
axis image
subplot(235)
plot(inliers(:, 1), inliers(:, 3), 'o')
axis image
subplot(236)
norm_metric = bb_cropped.norms(:, 3) > -0.4; 
%imagesc(+reshape(norm_metric, size(bb_cropped.grey)) + bb_cropped.mask)
imagesc(reshape(norm_metric, size(bb_cropped.grey)) + 2*bb_cropped.mask)
%imagesc(bb_cropped.mask)
axis image

%% plotting spider

%% displayin the spider
for ii = 1:12
    subplot(4, 3, ii)
    imagesc(bb_cropped.spider(:, :, ii+12))
    axis image
    colormap(jet)
    colorbar
end



