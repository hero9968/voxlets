function R = scale_matrix(scale)
% forms a 3x3 scaling matrix designed to be used in homodeneous coodinates

R = [scale, 0, 0;
    0, scale, 0;
    0, 0, 1];