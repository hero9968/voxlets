function XY_out = apply_transformation_2d(XY, transformation, transform_type)
% apply a 3x3 transformation matrix to 2d points
% affine way for speed ? projective way for full transform

% input check
assert(size(XY, 1)==2);
assert(size(transformation, 1) == 3 && size(transformation, 2) == 3);

if strcmp(transform_type, 'affine')
    
    % extracting rotation and translation parts from matrix
    T_rot = transformation(1:2, 1:2);
    T_trans = transformation(1:2, 3);

    % apply rotation
    XY_rot = T_rot * XY;

    % apply transformation
    XY_out = [XY_rot(1, :) + T_trans(1); XY_rot(2, :) + T_trans(2)];
    
elseif strcmp(transformation_type, 'projective')

    % convert to homogeneous coordinates
    XY_hom = cart2hom(XY);

    % apply transformation
    XY_trans = (transformation * XY_hom);

    % convert back to cartesian
    XY_out = hom2cart(XY_trans);

end

% output check
assert(all(size(XY_out)==size(XY)));