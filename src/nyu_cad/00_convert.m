
% convert meshes
base_folder = '../../data/cleaned_3D/mat/'


function [verts, faces] = combine_mesh(verts, new_verts, faces, new_faces)

    if numel(new_faces) > 0
        new_faces -= min(new_faces(:));
        new_faces += 1;
        new_faces += size(verts, 1);

        faces = [faces; new_faces];
        verts = [verts; new_verts];
    end
end


function [verts, faces] = combine_components(comp)

    verts = [];
    faces = [];
    for comp_idx = 1:length(comp(:))

        these_faces = cast(comp{comp_idx}.faces, 'int32');
        these_faces = fliplr(these_faces);
        [verts, faces] = combine_mesh(verts, comp{comp_idx}.vertices, faces, these_faces);
    end
end


function write_obj(fname, verts, faces)

    fid = fopen(fname, 'w');
    for vert_idx = 1:size(verts, 1)
        fprintf(fid, 'v %2.4f %2.4f %2.4f\n', verts(vert_idx, 1), verts(vert_idx, 2), verts(vert_idx, 3));
    end
    for f_idx = 1:size(faces, 1)
        fprintf(fid, 'f %d %d %d\n', faces(f_idx, 1), faces(f_idx, 2), faces(f_idx, 3));
    end
    fclose(fid);
end


function process_mat_file(base_folder, matname, remove_room_structure, binvox_save_folder)

    binvox_convert = true;

    matname
    binvox_save_name = [binvox_save_folder, matname(1:end-3), 'binvox'];
    obj_save_name = [binvox_save_folder, matname(1:end-3), 'obj']

    if exist(binvox_save_name, 'file') && exist(obj_save_name, 'file')
        ['Skipping ', binvox_save_name]
        return
    end

    mat = load([base_folder, matname]);

    model = mat.model;

    % write the camera details to a nice file for python to read or something
    % model.camera
    % model.camera.K'(:)
    % sdsds

    scene_verts = [];
    scene_faces = [];

    % loop over each object in mesh...
    for obj_idx = 1:size(model.objects, 2)

        % don't add the big planes, but everything else, add into the mesh
        this_label = model.objects{1, obj_idx}.model.label;
        if ~ismember(this_label, {'wall', 'floor', 'ceiling'}) || ~remove_room_structure

            % get the mesh
            comp = model.objects{1, obj_idx}.mesh.comp;
            [verts, faces] = combine_components(comp);

            % combine with everything else...
            [scene_verts, scene_faces] = combine_mesh(scene_verts, verts, scene_faces, faces);
        end
    end

    write_obj('/tmp/temp.obj', scene_verts, scene_faces);

    if binvox_convert
        % convert the tempory file
        [status, cmdout] = system(['binvox /tmp/temp.obj']);

        % move the new file
        movefile('/tmp/temp.binvox', binvox_save_name);
    end

    % move the obj file
    obj_save_name
    movefile('/tmp/temp.obj', obj_save_name);

end


listing = dir([base_folder, '*.mat'])

binvox_save_folders = {'../../data/cleaned_3D/binvox_no_walls/', '../../data/cleaned_3D/binvox_with_walls/'};
remove_room_structures = [true, false];

for type = [1] %, 2]
  for file_idx = 1:length(listing)
      temp = listing(file_idx).name(1:2);

      process_mat_file(base_folder, listing(file_idx).name, remove_room_structures(type), binvox_save_folders{type});
      % if strcmp(temp, '1_')
        % break
      % end
  end
end
