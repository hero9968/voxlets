function R = translation_matrix(x_diff, y_diff, z_diff)
% forms a 3x3 translation matrix designed to be used in homodeneous coodinates

R = [1, 0, 0, x_diff;
    0, 1, 0, y_diff;
    0, 0, 1, z_diff;
    0, 0, 0, 1];