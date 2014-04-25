function XY_out = apply_transformation_2d(XY, transformation)
% apply a 3x3 transformation matrix to 2d points
% no longer using homogeneous coordinates as this way is much quicker

% input check
assert(size(XY, 1)==2);

% extracting rotation and translation parts from matrix
T_rot = transformation(1:2, 1:2);
T_trans = transformation(1:2, 3);

% apply rotation
XY_rot = T_rot * XY;

% apply transformation
XY_out = [XY_rot(1, :) + T_trans(1); XY_rot(2, :) + T_trans(2)];

% output check
assert(all(size(XY_out)==size(XY)));