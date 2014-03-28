% a script to train a model from the training data...

clear
cd ~/projects/shape_sharing/2D/src
run('../define_params')
addpath('predict', genpath('utils'))
cd ~/projects/shape_sharing/2D/src

%% loading in all depths and shapes from disk...
load(paths.train_data, 'train_data')
load(paths.test_data, 'test_data')

%% now compute the model
run ../define_params
model = train_fitting_model(train_data.images, train_data.depths, params);
model.images = train_data.images;
model.depths = train_data.depths;
all_dists = cell2mat(model.shape_dists);
imagesc(all_dists)
num = 10

%%

%%
%close
clf
%num = 60;
num = num+1;
for ii = 1:3
    subplot(1, 3,ii); 
    combine_mask_and_depth(test_data.images{num}, test_data.depths{num})
    width = length(test_data.depths{num})
    set(gca, 'xlim', round([-width/2, 1.5*width]));
    set(gca, 'ylim',round([-width/2, 1.5*width]));
end

test_fitting_model(model, test_data.depths{num}, params)



%% fixing the rotating bug
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





