function M_out = transformation_matrix_from_vector(v_in, homogeneous)
% function to turn a vector into a rotation matrix, by sampling two 
% new random orthogonal vectors

assert(all(size(v_in) == [1, 3]))

v1 = normalise_length(v_in);

if dot(v1, [1,0,0]) < 0.999
    rand1 = [1, 0, 0];
else
    rand1 = normalise_length(rand(1, 3));
end

v2 = normalise_length(cross(rand1, v1));

v3 = normalise_length(cross(v1, v2));

M_out = [v1; v2; v3];

if nargin > 1 && homogeneous == 1

	M_out = [[M_out; 0, 0, 0], [0, 0, 0, 1]'];


end

