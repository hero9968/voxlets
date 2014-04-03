function T_out = icp_wrapper(model, data, T_init, outlier_dist)
% a wrapper for the icp mex funfction with som additinal functionality
plotting = 0;

assert(exist('icpMex', 'file')==3, 'Cannot find icpMex on path')
assert(size(data, 1) == 2);
assert(size(model, 1) == 2);


data(:, any(isnan(data), 1)) = [];

% performing ICP to refine alignment
try
    t_data_XY = apply_transformation_2d(data, T_init);
    temp_icp = icpMex(model, t_data_XY, eye(3), outlier_dist, 'point_to_plane');
    T_out = temp_icp * T_init;
catch err
    %keyboard
    disp(err)
    warning('ICP failed - not doing ICP step');
    T_out = T_init;
end
%T_out = T_init;


if cond(T_out) > 1e7
    warning(['Seems like conditioning is bad - using T_init instead of ICP result'])
    %keyboard
    T_out = T_init;
end


if plotting
    t_XY = apply_transformation_2d(data, T_out);
    plot(t_data_XY(1, :), t_data_XY(2, :)); 
    axis image; hold on; 
    plot(model(1, :), model(2, :), 'r'); 
    plot(t_XY(1, :), t_XY(2, :), 'g'); 
    hold off
    drawnow; 
    
    %pause(2)
end