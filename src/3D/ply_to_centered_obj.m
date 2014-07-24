
cd ~/projects/shape_sharing/3D/
clear
run define_params_3d.m
addpath(genpath('.'))
%edit write_obj

%%
for ii = params.files_to_use
    
    input_file = [paths.basis_models.originals '/' params.model_filelist{ii} '.obj'];
    output_file = [paths.basis_models.centred '/' params.model_filelist{ii} '.obj'];
    meta_file = [paths.basis_models.centred '/' params.model_filelist{ii} '.mat'];
    
    if exist(output_file,'file') && exist(meta_file,'file') && ~params.overwrite
        continue;
    end
	
    % reading in the object from disk
    [obj] = read_obj_simple(input_file);
    
    % getting an AABB
    aabb.min_vertex = min(obj.vertices);
    aabb.max_vertex = max(obj.vertices);
    
    % rationalise the file ? remove the redundent vertices
    n_obj = obj;
    [n_obj.vertices, m, n] = unique(n_obj.vertices, 'rows');
    n_obj.faces = n(n_obj.faces);
    
    % moving to be centered at (xy = 00)
    aabb.obj_centre = -(aabb.min_vertex + aabb.max_vertex) / 2;
    n_obj.vertices = apply_transformation_3d(n_obj.vertices, aabb.obj_centre);
    
    % rescaling object to have a diagonal AABB size of 1
    aabb.diag = sqrt(sum(range(obj.vertices).^2));
    n_obj.vertices = n_obj.vertices / aabb.diag;
    
    % saving in new path as an obj file
    write_obj_simple(n_obj, output_file);
    save(meta_file, 'aabb');
    
    %done(ii, length(ply_consts.model_names));
    ii
    
end

%%


%%
fid = fopen(input_file);
%A = textscan(fid, '%c %f %f %f', 'Delimiter', '\n', 'CommentStyle', '#', 'HeaderLines', 4);

fclose(fid)

