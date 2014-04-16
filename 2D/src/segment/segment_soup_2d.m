function binary_idxs = segment_soup_2d(depth, opts)
% segments a 2d depth measurement into a variety of different sements, each
% of which is represented as a row in a matrix

% setup and input checks
assert(isvector(depth));

thresholds = opts.thresholds;
nms_width = opts.nms_width;
max_segments = opts.max_segments;

% form matrix to be filled
idxs = nan(length(thresholds), length(depth));
%total_segments = 0;

% segment for each different distance threshold
for ii = 1:length(thresholds)
    
    this_threshold = thresholds(ii);
    idxs(ii, :) = segment_2d(depth, this_threshold, nms_width);
 
end

% removing non-unique segments
idxs(any(isnan(idxs), 2), :) = [];
idxs = unique(idxs, 'rows');

% now converting to binary and filtering
binary_idxs = filter_segments(idxs, opts);

% finally only returning the 'best' segments - in this case best is bigger!
num_segments = size(binary_idxs, 1);
final_number_segments = min(max_segments, num_segments);

segment_sizes = sum(binary_idxs, 2);
[~, sort_idx] = sort(segment_sizes, 'descend');

to_keep = sort_idx(1:final_number_segments);
binary_idxs = binary_idxs(to_keep, :);





