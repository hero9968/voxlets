function angle_hist = edge_angle_fv(mask, num_bins)
% computes the edge angle FV for the mask provided

mask = imclose(mask, strel('disk', 10));
C = centroid(mask);
B = bwboundaries(mask);

% find largest component
[~, max_idx] = max(cellfun(@length, B));
XY = (B{max_idx});
XY = fliplr(XY);

diffs = XY - repmat(C, size(XY, 1), 1);
dists = sqrt(diffs(:, 1).^2 + diffs(:, 2).^2);
all_angles = mod(atan2(diffs(:, 1), diffs(:, 2)), 2*pi);

%num_bins = 50;
bin_indices = round(all_angles/(2*pi) * num_bins)+1;
angle_hist = accumarray(bin_indices, dists, [num_bins+1, 1], @median);
angle_hist = angle_hist / sum(angle_hist);