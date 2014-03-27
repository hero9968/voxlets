% a script to train a model from the training data...

clear
cd ~/projects/shape_sharing/2D/src
run('../define_params')
addpath('predict', genpath('utils'))
cd ~/projects/shape_sharing/2D/src

%% loading in all depths and shapes from disk...
load(paths.train_data, 'train_data')

%% now compute the model
run ../define_params
model = train_fitting_model(train_data.images, train_data.depths, params);
model.images = train_data.images;
model.depths = train_data.depths;
all_dists = cell2mat(model.shape_dists);
imagesc(all_dists)
num = 10
%%
%close
clf
%num = num+1;
for ii = 1:3
    subplot(1, 3,ii); 
    combine_mask_and_depth(model.images{num}, model.depths{num})
    set(gca, 'xlim', [-50, 150]);
    set(gca, 'ylim', [-50, 200]);
end

test_fitting_model(model, train_data.depths{num}, params)





