%% script to generate halo and probably save files to disk...
clear
cd ~/projects/shape_sharing/src/3D
run define_params_3d.m
close all
addpath src/model_render/
addpath src/plotting/
addpath src/utils/
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

    % creating rotation matrix (proper way - but the renders are done the old way!)
%     v_to_O = normalise_length(T);
%     v_up = [0, 0, 1];
%     
%     if abs(dot(v_to_O, v_up))==1
%         r1 = [1, 0, 0];
%         r2 = [0, 1, 0];
%         r3 = v_to_O;    
%     else
%         r1 = normalise_length(cross(v_to_O, v_up));
%         r2 = normalise_length(cross(r1, v_to_O));
%         r3 = v_to_O;    
%     end

    % creating rotation matrix (old way)
    temp_null = null(T);
    temp_cross = cross(temp_null(:, 1)', temp_null(:, 2)')';
    if T(1) < 0
       temp_cross = -temp_cross;
       temp_null(:, 2) = -temp_null(:, 2);
    end
    R = [temp_null, temp_cross];

    % enforcing det = 1
    %R = [-r1; -r2; r3]';
    if abs(det(R) + 1) < 0.0001
        R(:, 2) = -R(:, 2);
    end
    det(R)

    % forming full transformation
    H = [R, T'; 0 0 0 1];

    if plotting 
        % plotting line on 3d plot
        T2 = R * [0, 0, 1]';
        T3 = R * [0, 1, 0]';
        T4 = R * [1, 0, 0]';
        hold on
        plot3(T(1), T(2), T(3), 'o')
        a = 0.5;
        line([T(1), T(1) + a*T2(1)], [T(2), T(2) + a*T2(2)], [T(3), T(3) + a*T2(3)]);
        line([T(1), T(1) + a*T3(1)], [T(2), T(2) + a*T3(2)], [T(3), T(3) + a*T3(3)], 'color', 'g');
        line([T(1), T(1) + a*T4(1)], [T(2), T(2) + a*T4(2)], [T(3), T(3) + a*T4(3)], 'color', 'r');
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
    ii;
    
end

