function mask_out = pca_symmetry_predict( depth_in, im_height)
% a function to predict the occupancy mask given the depth image in
% this function will use a very simple symmetry predictor

width = length(depth_in);
height = im_height;

% representing edge of object as set of points in 2D
X = 1:width;
Y = depth_in;
XY_original = double([X', Y']);
XY_original(any(isnan(XY_original), 2), :) = [];

[A, B] = eig(cov(XY_original));

X2 = (-width/2):0.01:(1.5*width);
Y2 = interp1(double(X), double(Y), X2, 'linear', 'extrap');

XY = double([X2', Y2']);


%todo  - here need to ensure that the 'normal' vector points away from the camera
%normal = A(:, 1);
%t_normal = normal ./ sqrt(sum(normal.^2));
%cos_angle = dot( [0, 1], t_normal );
if A(2, 1) > 0
    A(:, 1) = -A(:, 1);
end

% projecting points onto vectors
x_orig_transformed = dot(XY_original', repmat(A(:, 1), 1, length(XY_original)));
x_transformed = dot(XY', repmat(A(:, 1), 1, length(XY)));
y_transformed = dot(XY', repmat(A(:, 2), 1, length(XY)));


    

% findin position of minimum in direction of least varience
min_pos = min(x_orig_transformed);

% creating flipped shape about this axis
new_x_transformed = min_pos - (x_transformed-min_pos);

% interpolate to the new points
%long_y = -width:(2*width);
%new_x_extrap = interp1(new_x_transformed(1, :), new_x_transformed(2, :), long_y, 'cubic', 'extrap');


% rotating new shape back into the original axes
transformed_shape = ([new_x_transformed; y_transformed]' * inv(A))';



if 0
    subplot(231); plot(X, Y); axis image
    subplot(232); 
    plot(x_transformed, y_transformed); 
    hold on;
    plot(new_x_transformed, y_transformed, 'r'); axis image
    hold off
    subplot(233); 
    plot(transformed_shape(1, :), transformed_shape(2, :), 'r'); 
    hold on
    plot(X, Y); 
    hold off
    axis image
end

% interpolating the new points
%Y_transformed = interp1(transformed_shape(1, :), transformed_shape(2, :), X, 'cubic', 'extrap');
%Y_transformed(isnan(Y_transformed)) = 1;
%Y_transformed(Y_transformed==0) = 1;
%Y_transformed = round(Y_transformed);
x_round = round(transformed_shape(1, :));
to_remove = x_round <=0 | x_round > width;
x_round(to_remove) = [];
transformed_shape(:, to_remove) = [];
Y_transformed = round(accumarray(x_round(:), transformed_shape(2, :), [width, 1], @max));
Y_transformed(Y_transformed>height) = height;

% filling in the final mask
grid1 = fill_grid_from_depth(Y, height, 1);
grid2 = fill_grid_from_depth(Y_transformed, height, 1);
object_edge = fill_grid_from_depth(Y, height, 0);

% constructing final grid
mask_out = (grid1-grid2) == 1;
mask_out(object_edge==1) = 1;
mask_out = single(mask_out);


if 0
    figure
    subplot(131); imagesc(grid1); axis image
    subplot(132); imagesc(grid2); axis image
    subplot(133); imagesc(mask_out); axis image
end



