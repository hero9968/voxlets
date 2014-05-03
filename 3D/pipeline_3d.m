clear
cd ~/projects/shape_sharing/3D
define_params_3d

addpath model_render/src/

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Basis shapes 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% generating the csv files which define the sphere to take pictures from
generate_halo

%% creating copies of the original obj files, centering and resizeing them, collapsing coincident vertices
% NB (took several hours when I last ran it!)
ply_to_centered_obj

%% rendering the objects from each point in the sphere. Each render is a separate mat file
% (Not yet run all of this!)
render_training_views

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Full test images
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
