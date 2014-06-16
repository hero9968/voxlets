function distance = average_neighbour_distance(XY_model, XY_data)
% average_neighbour_distance(XY_model, XY_data)
% 
% for each point in XY_model, finds the cloest point in XY_data. Returns
% average of all these points


% getting the ground truth points

% getting the predicted points
to_remove = any(isnan(XY_model), 1);
XY_model(:, to_remove) = [];

% doing some kind of distance matrix
T = pdist2(double(XY_model)', double(XY_data)');
dists = min(T, [], 2);

% make robust
%dists(dists>10) = 10;

distance = mean(dists.^2);

if isempty(distance)
    distance = inf;
end
    
    