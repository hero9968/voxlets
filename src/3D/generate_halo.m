%% script to generate halo and probably save files to disk...
clear
cd ~/projects/shape_sharing/src/3D
run define_params_3d.m
close all
addpath model_render/src/
addpath plotting/
plotting = 1;

if ~exist(paths.basis_models.halo_path, 'dir')
    mkdir(paths.basis_models.halo_path)
end

%%

% setting the intrinsic matrix (just for the saving)
K = params.half_intrinsics;

% setting parameters for the halo creation
level = params.model_render.level;
radius = params.model_render.radius;

% generating the halo
xyz = radius * icosahedron2sphere(level);

if params.model_render.hemisphere
    % removing points below z = 0 plane
    to_remove = xyz(:, 3) <=0;
    xyz(to_remove, :) = [];
end

% applying some kind of ordering to the points
xyz = sortrows(xyz, [3, 1, 2]);

if plotting
    plot3d(xyz)
end

for ii = 1:size(xyz, 1)

    % extracting translation
    T = xyz(ii, :);

    % creating rotation matrix
    temp_null = null(T);
    temp_cross = cross(temp_null(:, 1)', temp_null(:, 2)')';
    if T(1) < 0
       temp_cross = -temp_cross;
       temp_null(:, 2) = -temp_null(:, 2);
    end
    R = [temp_null, temp_cross];
    det(R)

    % forming full transformation
    H = [R, T'; 0 0 0 1];

    if plotting 
        % plotting line on 3d plot
        T2 = R * [0, 0, 1]';
        hold on
        line([T(1), T(1) + T2(1)], [T(2), T(2) + T2(2)], [T(3), T(3) + T2(3)]);
        hold off
    end
    
    all_transforms(:, :, ii) = H;

    % saving transformation
    csv_name = sprintf(paths.basis_models.halo_file, ii);
    csvwrite(csv_name, H);
    
    R = H';
    mat_name = sprintf(paths.basis_models.halo_file_mat, ii);
    save(mat_name, 'R', 'K')
    % done(ii)
    ii
    
end

