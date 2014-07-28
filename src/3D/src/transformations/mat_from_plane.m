function T_out = mat_from_plane(plane_eqn)
% finds a homogenerous transformation to rotate into the 
% coordinates defined by a 1x4 plane eqn

assert(isvector(plane_eqn) && length(plane_eqn)==4)

scaling = sqrt(sum(plane_eqn(1:3).^2));

% forming the rotation matrix
r3 = normalise_length(plane_eqn(1:3));
r2 = normalise_length(cross([0, 0, 1], r3));
r1 = normalise_length(cross(r2, r3));
rot = [r1; r2; r3];

assert(abs(det(rot)-1) < 0.0001);

% forming final transformation
T2 = [rot, [0; 0; 0]; 0 0 0 1];
T1 = [eye(3), plane_eqn(4) * [0, 0, 1]' / scaling; 0 0 0 1];
T_out = T1*T2;

