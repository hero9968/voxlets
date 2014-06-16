function [XY, norms, angle_hist] = edge_normals(mask, k)
% computes ordered set of points on edge of binary mask, plus their normals
% (ensuring the normals face away from the mask center)
% k is half window width for normal computation

B = bwboundaries(mask);

% find largest component
[~, max_idx] = max(cellfun(@length, B));
XY = (B{max_idx});

% now compute the normals on the boundary
N = size(XY, 1);
norms = nan(N, 2);

% loop over each point
for ii = 1:N
    
    % extracting neighbouring points
    nn = mod((ii-k:ii+k)-1, N) + 1;
    t_XY = XY(nn, :);
   
    % computing normals
    [rot, dummy] = eig(cov(t_XY));
    idx = diag(dummy)==min(diag(dummy));
    norms(ii, :) = rot(:, idx(1))';
    curve(ii) = min(diag(dummy));
       
    % sorting out direciton
    tt_XY = round(XY(ii, :) + (rot * [3; 0])');
    if mask(tt_XY(1), tt_XY(2))
        norms(ii, :) = -norms(ii, :);
    end
    
end

XY = fliplr(XY);
norms = fliplr(norms);

if nargout == 3
    
    C = centroid(mask);
    
    diffs = XY - repmat(C, size(XY, 1), 1);
    dists = sqrt(diffs(:, 1).^2 + diffs(:, 2).^2);
    all_angles = mod(atan2(diffs(:, 1), diffs(:, 2)), 2*pi);
    
    num_bins = 50;
    bin_indices = round(all_angles/(2*pi) * num_bins)+1;
    angle_hist = accumarray(bin_indices, dists, [num_bins+1, 1], @median);
    angle_hist = angle_hist / sum(angle_hist);
    
end

