 function norms = normals_radius_2d( XY, radius )
% compute normals for set of 2D points

N = size(XY, 2);
assert(size(XY, 1) == 2);

norms = nan(2, N);
dists = squareform(pdist(XY'));


for ii = 1:N
    
    % extracting local points
    these_dists = dists(ii, :);
    inlier_idx = these_dists < radius;
    sum(inlier_idx);
    
    t_XY = XY(:, inlier_idx);
    
    % getting normal
    if size(t_XY, 2) > 2
        [rot, dummy] = eig(cov(t_XY'));
        idx = diag(dummy)==min(diag(dummy));
        normal = rot(:, idx(1));
    else
        normal = [0, -1];
    end

    % fixing for normal in wrong direction
    if normal(2) > 0
        normal = -normal;
    end
     
    % filling in
    norms(:, ii) = normal;
    
end