%% script to generate halo and probably save files to disk...
clear
run ../define_params_3d.m
close all

%%

% setting parameters for the halo creation
level = 2;
radius = 4;

% generating the halo
xyz = radius * icosahedron2sphere(level);

if hemisphere
    % removing points below z = 0 plane
    to_remove = xyz(:, 3) <=0;
    xyz(to_remove, :) = [];
end

% applying some kind of ordering to the points
xyz = sortrows(xyz, [3, 1, 2]);


plot3d(xyz)


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
   
   % plotting line on 3d plot
   T2 = R * [0, 0, 1]';
   line([T(1), T(1) + T2(1)], [T(2), T(2) + T2(2)], [T(3), T(3) + T2(3)]);
   hold on
   
   all_transforms(:, :, ii) = H;
   
   % saving transformation
   csv_name = sprintf(paths.basis_models.halo, ii);
   %csvwrite(csv_name, H);
   % done(ii)
   ii
    
end
hold off

%% creating very simple rotations for initial testing
%{
count = 1;
for pos = 1:3
    for neg = [-1, 1]
        T = [0, 0, 0];
        T(pos) = neg * 3;
        R = eye(3);
        
        H = [R, T'; 0 0 0 1];
        
        csv_name = sprintf('/Users/Michael/Data/Others_data/google_warehouse/rotations/mat_%d.csv', count);
        count = count + 1;
        csvwrite(csv_name, H);
    end
end
%}

%% plotting the depth images created
close all
clear temp
for ii = 1:16
    filename = sprintf('/Users/Michael/Data/Others_data/google_warehouse/obj_beds/BED/depths/depth_%d.mat', ii);
    
    temp(ii) = load(filename);
    temp(ii).depth( abs(temp(ii).depth - 10) < 0.001 ) = nan;
    
    
    subplot(4, 4, ii);
    imagesc2(temp(ii).depth);
    axis on
    
end


%% now want to try to get the point clouds out and realign using the matrices I have...
close all
clear temp
cols = 'ymcrgbwkymcrgbwkymcrgbwkymcrgbwkymcrgbwk';

for ii = 1:16
    
    filename = sprintf('/Users/Michael/Data/Others_data/google_warehouse/obj_beds/BED/depths/depth_%d.mat', ii);
    
    % loading depth
    T = load(filename);
    temp(ii).depth = T.depth;
    temp(ii).depth( abs(temp(ii).depth - 10) < 0.001 ) = nan;

    % converting to XYZ
    temp(ii).xyz = 1000*depth2xyz(temp(ii).depth);
    
    % now transforming using the transformation
    %trans = pinv(all_transforms(:, :, ii));
    %trans(2, 4) = -trans(2, 4);
    %trans = eye(4);
    if 1
        
        trans = pinv(all_transforms(:, :, ii))
        %trans(1:3, 3) = -trans(1:3, 3)
        %RR = (trans(1:3, 1:3))';
        %RR = (all_transforms(1:3, 1:3, ii));
        RR_inv = trans(1:3, 1:3);
        %tttemp = RR_inv(:, 3);
        %RR_inv(:, 3) = RR_inv(:, 2);
        %RR_inv(:, 2) = tttemp;
        
        TT = trans(1:3, 4);

        temp(ii).xyz_trans = apply_transformation_3d(temp(ii).xyz, TT);
        temp(ii).xyz_trans = apply_transformation_3d(temp(ii).xyz_trans, RR_inv);
        %temp(ii).xyz_trans = apply_transformation_3d(temp(ii).xyz_trans, RR);
        %temp(ii).xyz_trans = apply_transformation_3d(temp(ii).xyz, pinv(RR));
        %temp(ii).xyz_trans = apply_transformation_3d(temp(ii).xyz_trans, pinv(trans));
        
        %temp(ii).xyz_trans = apply_transformation_3d(temp(ii).xyz_trans,);
        
    else
        temp(ii).xyz_trans = temp(ii).xyz;
    end
    %subplot(4, 4, ii);
    %imagesc2(temp(ii).depth);
    plot3d(temp(ii).xyz_trans, cols(ii));
    hold on
end
hold off




%D = reshape(depth', [480, 640]);
%D = depth;
%imagesc2(T.depth)
%whos d*