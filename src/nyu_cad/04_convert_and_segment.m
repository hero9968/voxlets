
% convert meshes
base_folder = '../../data/cleaned_3D/mat/'
save_folder = '../../data/nyu_renders_split/'


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


function process_mat_file(base_folder, matname, remove_room_structure, obj_save_folder_base)

    binvox_convert = true;

    matname
    % binvox_save_name = [binvox_save_folder, matname(1:end-3), 'binvox'];
    obj_save_dir = [obj_save_folder_base, matname(1:end-4)]

    if ~exist(obj_save_dir, 'dir')
        mkdir(obj_save_dir)
    end

    mat = load([base_folder, matname]);

    model = mat.model;

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

            % save this mesh separately
            savepath = sprintf([obj_save_dir, '/mesh_%04d.obj'], obj_idx)
            write_obj(savepath, verts, faces);

            %% combine with everything else...
            % [scene_verts, scene_faces] = combine_mesh(scene_verts, verts, scene_faces, faces);
        end
    end

end


listing = dir([base_folder, '*.mat'])

binvox_save_folders = {'../../data/cleaned_3D/binvox_no_walls/', '../../data/cleaned_3D/binvox_with_walls/'};
remove_room_structures = [true, false];

for type = [1] %, 2]
  for file_idx = 1:length(listing)
      temp = listing(file_idx).name(1:2);

      process_mat_file(base_folder, listing(file_idx).name, remove_room_structures(type), save_folder);
      % if strcmp(temp, '1_')
        % break
      % end
  end
end
