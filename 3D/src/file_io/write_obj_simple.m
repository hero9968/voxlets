function obj = write_obj_simple(obj, filename)
% function to read an object file from disk

fid = fopen(filename, 'w');

fprintf(fid, '# Converted by mfirman with a very simple MATLAB function write_obj_simple.m\n');

% writing vertices
for ii = 1:size(obj.vertices, 1)
    fprintf(fid, 'v %.3f %.3f %.3f\n', obj.vertices(ii, 1), obj.vertices(ii, 2), obj.vertices(ii, 3));
end

% writing faces
for ii = 1:size(obj.faces, 1)
    fprintf(fid, 'f %d %d %d\n', obj.faces(ii, 1), obj.faces(ii, 2), obj.faces(ii, 3));
end

fclose(fid);
