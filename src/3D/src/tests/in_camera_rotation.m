% input point
v_in = [1, 1, 0.5];
v_in = normalise_length(v_in);

clc

%% given a point in 3D space, find the vector from the point to the origin
v_to_O = -v_in;
%null(v_in

%construct the upwards rotation matrix
v_up = [0, 0, 1];
r1 = normalise_length(cross(v_to_O, v_up));
r2 = normalise_length(cross(r1, v_to_O));
r3 = v_to_O;
R = [r1; r2; r3];
if abs(det(R) + 1) < 0.0001
    R = [r1; -r2; r3];
end
R
det(R)

%%
temp_null = null(v_to_O);
%temp_cross = cross(temp_null(:, 1)', temp_null(:, 2)')';
temp_cross = v_to_O';
R_orig = [temp_null, temp_cross];
if abs(det(R_orig) + 1) < 0.0001
    R_orig = [-temp_null(:, 1), temp_null(:, 2), temp_cross];
end
R_orig
det(R_orig)

%%
diff_mat = R * R_orig;
acosd(diff_mat(1))
asind(diff_mat(2))
-atan2d(diff_mat(2), diff_mat(1))


%%
R * inv(R_orig)
R_small = R_orig(2:3, 1:2)
acosd(R_small(1))