% combine depth images (and probably normals) into one file
% I'll crop the images, and include a 'top-left' field.
% Will be interesting to see difference in disk space...

cd ~/projects/shape_sharing/3D/
clear
addpath(genpath('.'))
addpath ../2D/src/utils/
run define_params_3d.m

number_renders = 42;

%%
for ii = 1%:length(params.model_filelist)
    
    % output place
    model = params.model_filelist{ii};
    outfile = sprintf(paths.basis_models.combined_file, model);
    
    if exist(outfile, 'file')
        disp(['Skipping ' num2str(ii)])
        continue
    end
    
    % setting up the variables to be filled
    clear combined c2

    % loop over each image and combine all the results together
    tic
    for jj = 1:number_renders

        this_depth_name = sprintf(paths.basis_models.rendered, model, jj);
        this_norms_name = sprintf(paths.basis_models.normals, model, jj);
        
        this_depth = load_for_parfor(this_depth_name, 'depth');
        max_depth = max(this_depth(:));
        mask = abs(this_depth-max_depth)< 0.001;
        this_depth(mask) = 0;
        [cropped_depth, top_left] = boxcrop_2d(this_depth);
               
        normals = load_for_parfor(this_norms_name, 'normals');
        % hack here for normals which were not stripped out correctly...
        if length(normals) == 240*320
            normals(mask, :) = [];
        end
        
        cropped(jj).cropped_depth = cropped_depth;
        cropped(jj).normals = single(normals);
        cropped(jj).top_left = top_left;
        
        uncropped(jj).depth = this_depth;
        uncropped(jj).norms = normals;
        
        t(jj) = dir(this_depth_name);
        
        
    end
    save('cropped_v7.mat', 'cropped')
    save('cropped_v6.mat', 'cropped', '-v6')
    save('uncropped_v7.mat', 'uncropped')
    save('uncropped_v6.mat', 'uncropped', '-v6')
    toc
    ii
end
names = {'cropped_v6.mat', 'cropped_v7.mat', 'uncropped_v6.mat', 'uncropped_v7.mat'};
sizes = [7.5, 5.2, 21, 5.9];

%% speed tests
num_tries = 1;

tic
for ii = 1:num_tries
    for jj = 1:42
        A = load('cropped_v6.mat');
        T = zeros(240, 320);
        [h, w] = size(A.cropped(jj).cropped_depth);
        T(A.cropped(jj).top_left(1):(A.cropped(jj).top_left(1)+h-1), A.cropped(jj).top_left(2):(A.cropped(jj).top_left(2)+w-1)) = A.cropped(jj).cropped_depth;
    end
end
time(1) = toc;

tic
for ii = 1:num_tries
    for jj = 1:42
        A = load('cropped_v7.mat');
        T = zeros(240, 320);
        [h, w] = size(A.cropped(jj).cropped_depth);
        T(A.cropped(jj).top_left(1):(A.cropped(jj).top_left(1)+h-1), A.cropped(jj).top_left(2):(A.cropped(jj).top_left(2)+w-1)) = A.cropped(jj).cropped_depth;
    end
end
time(2) = toc;

tic
for ii = 1:num_tries
    A = load('uncropped_v6.mat');
end
time(3) = toc;

tic
for ii = 1:num_tries
    A = load('uncropped_v7.mat');
end
time(4) = toc;

%%
subplot(121)
plot(time, sizes, 'o')
xlabel('Time')
ylabel('Size')
hold on
for ii = 1:4
    text(time(ii), sizes(ii), names{ii});
end
hold off