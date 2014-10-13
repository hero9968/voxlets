function [fv, dists_original, angle_original] = edge_shape_dists_norms(mask, dict)
% need to do some sort of rescaling...
% perhaps similar to in the 3D case?

num_samples = 20000;

[XY, norms] = edge_normals(mask, 15);

XY = XY *  normalise_scale_2d(XY');

[fv, dists_original, angle_original] = shape_dist_2d_dict(XY', norms', num_samples, dict);