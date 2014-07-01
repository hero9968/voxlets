function R = translation_matrix_3d(xyz_diff)
% forms a 3x3 translation matrix designed to be used in homodeneous coodinates

R = [1, 0, 0, xyz_diff(1);
    0, 1, 0, xyz_diff(2);
    0, 0, 1, xyz_diff(3);
    0, 0, 0, 1];