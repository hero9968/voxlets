function binary_segments = filter_segments(segments, opts)
% the idea is to filter the segmentation given in segmentsto remove (merge)
% very similar segments, remove small segments, etc.

min_size = opts.min_size;
overlap_threshold = opts.overlap_threshold;

binary_segments = segments_to_binary(segments);

% remove segments which are too small
segment_sizes = sum(binary_segments, 2);
to_remove = segment_sizes < min_size;
binary_segments(to_remove, :) = [];

% merge/remove overlapping segments
distances = pdist(binary_segments, @overlap_function);
distances = triu(squareform(distances));
to_remove = any(distances > overlap_threshold, 1);
binary_segments(to_remove, :) = [];



function D = overlap_function(a, b)

a_rep = repmat(a, size(b, 1), 1);

intersection = sum( a_rep & b, 2 );
union = sum( a_rep | b, 2 );

D = intersection ./ union;
D = D(:);
assert(length(D) == size(b, 1))