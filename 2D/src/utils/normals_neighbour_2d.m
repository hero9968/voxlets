 function norms = normals_neighbour_2d( XY, hww, threshold )
% compute normals for set of 2D points
% hww is half window width for finding neighbours
% threshold is optional distance threshold for inclusion

N = size(XY, 2);
assert(size(XY, 1) == 2);

norms = nan(2, N);

for ii = 1:N
    
    % extracting local points
	inlier_idx = (ii-hww):(ii+hww);
    inlier_idx(inlier_idx<1) = [];
    inlier_idx(inlier_idx>N) = [];
    
    t_XY = XY(:, inlier_idx);
    
    if nargin == 3
        dists = pdist2(XY(:, ii)', t_XY');
        t_XY(:, dists > threshold) = [];
    end
    
    % getting normal
    if size(t_XY, 2) > 1
        [rot, dummy] = eig(cov(t_XY'));
        idx = diag(dummy)==min(diag(dummy));
        normal = rot(:, idx);
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