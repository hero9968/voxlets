function obj = read_obj_simple(input_file)
% function to read a .obj file from disk, in a really simple way
% this function strips all info like materials, colours and textures, 
% and names of parts etc.


% open file for reading
fid = fopen(input_file);
all_file = textscan(fid, '%s', 'Delimiter', '\n', 'CommentStyle', '#', 'HeaderLines', 4);
fclose(fid);

% initialise the arrays to be a guess at the maximum size
%vertex_cell = cell(1, 5e5);
N = length(all_file{1});
obj.vertices = nan(N, 3);
obj.faces = nan(N, 3);


% counters
v_count = 1;
f_count = 1;

% keep reading while the file is good
for ii = 1:N
    
    this_line = all_file{1}{ii};
    
    % switch on the line identifier
    switch this_line(1:2)
        
        case 'v '
            %new_vertex = str2num(tline(3:end));
            new_vertex = sscanf(this_line(3:end), '%f')';
            obj.vertices(v_count, :) = new_vertex;
            %vertex_cell{v_count} = tline(3:end);
            v_count = v_count + 1;
            
        case 'f '
            %new_face = str2num(tline(3:end));
            new_face = sscanf(this_line(3:end), '%d')';
            obj.faces(f_count, :) = new_face;
            f_count = f_count + 1;
    end

end


% clean up
obj.vertices(any(isnan(obj.vertices), 2), :) = [];
obj.faces(any(isnan(obj.faces), 2), :) = [];