function cloud = loadpgm_as_cloud(filename, intrinsics)


%clear cloud
%filepath = '/Users/Michael/data/others_data/ECCV_dataset/pcd_files/frame_20111220T111153.549117.pcd';
%P = loadpcd(filepath);
%cloud.xyz = P(:, :, 1:3);
%cloud.xyz = reshape(permute(cloud.xyz, [3, 1, 2]), 3, [])';
cloud.depth = readpgm(filename);


cloud.xyz = reproject_depth(cloud.depth, intrinsics);

% inserting nans
nan_locations = cloud.depth==0;
cloud.xyz(nan_locations(:), :) = nan;
cloud.depth(nan_locations) = nan;

% synthesising an RGB channel
cloud.rgb = repmat(cloud.depth, [1, 1, 3])/2;
