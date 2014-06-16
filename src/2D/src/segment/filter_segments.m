function [binary_segments2, binary_origins3] = filter_segments(segments, opts)
% the idea is to filter the segmentation given in segmentsto remove (merge)
% very similar segments, remove small segments, etc.

% remove the nans
%if any(isnan(segments(:)))
%    error('Cannot have nans at the moment...')
%end
num_points = size(segments, 2);
num_segments_in = size(segments, 1);
nan_points = any(isnan(segments), 1);
segments(:, nan_points) = [];

% input options
min_size = opts.min_size;
overlap_threshold = opts.overlap_threshold;

% convert the segments to binary
[binary_segments, binary_origins] = segments_to_binary(segments, -1);

% remove segments which are too small
segment_sizes = sum(binary_segments, 2);
segments_to_remove = segment_sizes < min_size;
binary_segments(segments_to_remove, :) = [];
binary_origins(segments_to_remove, :) = [];

% take unique binary segments (doing this first helps with speed as unique
% is faster than pdist)
if 0 
    [binary_segments, ~, m] = unique(binary_segments, 'rows');

    % sort out the binary table following the `unique' operation
    for ii = 1:size(binary_segments, 1)
        inliers = m == ii;
        inlier_rows = binary_origins(inliers, :);
        binary_origins2(ii, :) = any(inlier_rows, 1);
    end
else
    binary_origins2 = binary_origins;
end

% merge/remove overlapping segments
overlaps = pdist(binary_segments, @overlap_function);
distances = 1 - overlaps;
Z = linkage(distances, 'single');
%T = 1:size(Z, 1); %cluster(Z, 'cutoff', overlap_threshold)
T = cluster(Z, 'cutoff', overlap_threshold, 'criterion', 'distance');
for ii = 1:max(T)
    inliers = T==ii;

    inlier_rows = binary_origins2(inliers, :);
    binary_origins3(ii, :) = any(inlier_rows, 1);

    inlier_binary = binary_segments(inliers, :);
    binary_segments2(ii, :) = mean(inlier_binary, 1);

end



function D = overlap_function(a, b)

a_rep = repmat(a, size(b, 1), 1);

intersection = sum( a_rep & b, 2 );
union = sum( a_rep | b, 2 );

D = intersection ./ union;
D = D(:);
assert(length(D) == size(b, 1))