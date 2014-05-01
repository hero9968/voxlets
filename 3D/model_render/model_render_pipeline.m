% 
clear
define_params_3d
params

%% generating the csv files which define the sphere to take pictures from
generate_halo

%% creating copies of the original obj files, centering and resizeing them, collapsing coincident vertices
ply_to_centered_obj

%% rendering the objects from each point in the sphere. Each render is a separate mat file
render_training_views
