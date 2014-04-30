function xyz_out = apply_transformation_3d(xyz_in, T)

assert(size(xyz_in, 2)==3);
N = size(xyz_in, 1);

if all(size(T)==[4, 4])
    
    % applying full homography
    xyz_in = [xyz_in, ones(N, 1)];
    temp = (T * xyz_in')';
    xyz_out = temp(:, 1:3) ./ repmat(temp(:, 4), 1, 3);
    
elseif isvector(T) && length(T) == 3
    
    % applying translation
    T = T(:)';
    xyz_out = xyz_in + repmat(T, N, 1);    
    
elseif all(size(T)==[3, 3])
    
    % applying rotation
    xyz_out = (T * xyz_in')';
    
end