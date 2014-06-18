function P = readpgm(filename)

fid = fopen(filename, 'r');
P = textscan(fid,'%f','headerlines',4);
fclose(fid);

P = flipud(reshape(P{1},[640, 480])');