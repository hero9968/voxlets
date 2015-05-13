% plotting the principle components of 2d shape
% used to help fixing the rotating bug
clear
cd ~/projects/shape_sharing/2D/src
run ../define_params
load(paths.test_data, 'test_data')
num = 1;

%%
clf
num = num+1;
depth = test_data.depths{num};
Y = (double(depth));
X = 1:length(Y);
[~, ~, this_transform_to_origin] = transformation_to_origin_2d(X, Y);
coordinate_frame = [0, 0, 10; 10, 0, 0];
coordinate_frame_trans = apply_transformation_2d(coordinate_frame, this_transform_to_origin);
plot(X, Y, 'o');
hold on
plot(coordinate_frame(1, :), coordinate_frame(2, :), 'r')
plot(coordinate_frame_trans(1, :), coordinate_frame_trans(2, :), 'r')
hold off
axis image
