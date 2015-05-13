function cloud = loadpcd_as_cloud(filepath)
% wrapper for loadpcd which makes the output into a nice structure

P = loadpcd(filepath);

cloud.xyz = P(:, :, 1:3);
cloud.xyz = reshape(permute(cloud.xyz, [3, 1, 2]), 3, [])';

cloud.depth = P(:, :, 3);

cloud.rgb = P(:, :, 4:6);


