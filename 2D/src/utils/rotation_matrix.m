function R = rot_matrix(angle_in_degrees)
% forms a 3x3 rotation matrix designed to be used in homodeneous coodinates

R = [cosd(angle_in_degrees), -sind(angle_in_degrees), 0;
    sind(angle_in_degrees), cosd(angle_in_degrees), 0;
    0, 0, 1];