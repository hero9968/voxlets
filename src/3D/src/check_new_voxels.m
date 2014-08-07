% script to check if both the voxelisations are good or not...

old_vox_T = [0 1/100 0 -0.5-0.0025; 
         	 1/100 0 0 -0.5-0.0025; 
             0 0 1/100 -0.5; 
             0 0 0 1];

new_vox_T = [0 1/150 0 -0.5-0.0025; 
         	 1/150 0 0 -0.5-0.0025; 
             0 0 1/150 -0.5; 
             0 0 0 1];
         
test_transform = [cosd(30), -sind(30), 0, 2;
                sind(30), cosd(30), 0, 2;
                0, 0, 1, 0;
                0, 0, 0, 1];

modelname = '2d971ef95b2164a0880cca8966535e8d';
old_vox_path = ['~/projects/shape_sharing/data/3D/basis_models/voxelised_text/' modelname '.txt'];
new_vox_path = ['~/projects/shape_sharing/data/3D/basis_models/voxelised/' modelname '.mat'];

old_vox_raw = importdata(old_vox_path, '', 1);
old_vox_raw = old_vox_raw.data;
new_vox_raw = load(new_vox_path);

old_size = [100, 100, 100];
old_vox_idx = full_3d(old_size, old_vox_raw);

new_size = [150, 150, 150];
new_vox_idx = full_3d(new_size, new_vox_raw.sparse_volume);

[old_full] = find(old_vox_idx);
[old.i, old.j, old.k] = ind2sub(old_size, old_full);
old_vox_world = apply_transformation_3d( [old.i, old.j, old.k], test_transform*old_vox_T);

[new_full] = find(new_vox_idx);
[new.i, new.j, new.k] = ind2sub(new_size, new_full);
new_vox_world = apply_transformation_3d( [new.i, new.j, new.k], test_transform*new_vox_T);


plot3d(old_vox_world, 'b')
hold on
plot3d(new_vox_world, 'r')
hold off
