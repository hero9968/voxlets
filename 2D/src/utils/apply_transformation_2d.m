function XY_out = apply_transformation_2d(XY, transformation)

% input check
assert(size(XY, 1)==2);

% convert to homogeneous coordinates
XY_hom = [XY; ones(1, size(XY, 2))];

% apply transformation
XY_rot = (transformation * XY_hom);

% convert back to cartesian
X_rot = XY_rot(1, :) ./ XY_rot(3, :);
Y_rot = XY_rot(2, :) ./ XY_rot(3, :);
XY_out = [X_rot; Y_rot];

% output check
assert(all(size(XY_out)==size(XY)));