function render_ply( object_number, H )


Ply_consts;
model_name = ply_consts.model_names{object_number}

if nargin == 1
    % generating a kind of random vieqw
    T = [2, 3, 2];
    %R3 = normalise_length(-T)';
    %R1 = normalise_length(cross(R3, [1, 0, 0]')')';
    %R2 = cross(R3, R1);
    %R = [R1, R2, R3];
    %H = [R, T'; 0 0 0 1]
    %det(R);
    %det(H);
    temp_null = null(T);
    temp_cross = cross(temp_null(:, 1)', temp_null(:, 2)')';
    if T(1) < 0
       temp_cross = -temp_cross;
       temp_null(:, 2) = -temp_null(:, 2);
    end
    R = [temp_null, temp_cross];
    det(R)

    % forming full transformation
    H = [R, T'; 0 0 0 1]
    
end

% write the transform to disk
%save('temp_transform.mat', 'T')
csvwrite('temp_transform.csv', H);

% call the python function
run_cmd = ['python python/render_specific_object.py ' model_name ' temp_transform.csv here.mat']
[A, B] = system(run_cmd)
