function render_ply( object_number, H )

%%
%Ply_consts;
%model_name = ply_consts.model_names{object_number}
%%
if nargin == 1
    
    % generating a kind of random vieqw
    T = [2, 3, 2];

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
%%
% write the transform to disk
%save('temp_transform.mat', 'T')
csvwrite('temp_transform.csv', H);
model_name = 'temp';
% call the python function
run_cmd = ['python render_specific_object.py ' model_name ' temp_transform.csv here.mat']
[A, B] = system(run_cmd)

%% loading in the render and viewing
A = load('here.mat')
imagesc(A.depth)
axis image