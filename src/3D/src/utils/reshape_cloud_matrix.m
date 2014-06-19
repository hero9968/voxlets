function c_out = reshape_cloud_matrix( c_in )
% reshapes n x m x 3 matrix into a n*m x 3 matrix

temp = permute(c_in, [3, 1, 2]);
c_out = temp(:, :)';