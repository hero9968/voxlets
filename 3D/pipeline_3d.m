clear
cd ~/projects/shape_sharing/3D
define_params_3d

addpath src/model_render/src/

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Basis shapes 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% generating the csv files which define the sphere to take pictures from
generate_halo

%% creating copies of the original obj files, centering and resizeing them, collapsing coincident vertices
% NB (took several hours when I last ran it!)
ply_to_centered_obj

%% rendering the objects from each point in the sphere. Each render is a separate mat file
render_training_views

%% Computing normals from the basis shapes
% won't bother with this for now ? will do it without the normals!
%compute_training_normals

%% Creating feature vector dictionary
generate_shape_dist_dict

%% Computing feature vectors from the rendered images
computing_feature_vectors

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Full test images
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


