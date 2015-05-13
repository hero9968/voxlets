function depth_image = readpgm(filename)
% Reads in a depth image from a .pgm file as outputted from blensor
% Probably (definitely) not compatible with any other type of pgm file!

fid = fopen(filename, 'r');

% reading header
fgetl(fid); % P2
fgetl(fid); % Blensor output
array_size = fscanf(fid, '%d %d\n', [1, 2]); % image size
fgetl(fid); % 65535

% reading file contents
file_contents = textscan(fid,'%f');
depth_image = flipud(reshape(file_contents{1}, array_size)');

fclose(fid);

