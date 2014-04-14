function idxs = segment_soup_2d(depth, params)
% segments a 2d depth measurement into a variety of different sements, each
% of which is represented as a row in a matrix

%thresholds = 5:5:40;

% setup and input checks
assert(isvector(depth));

thresholds = params.segment_soup.thresholds;
nms_width = params.segment_soup.nms_width;
max_segments = params.segment_soup.max_segments;

% form matrix to be filled
idxs = nan(length(thresholds), length(depth));
total_segments = 0;

% segment for each different distance threshold
for ii = 1:length(thresholds)
    
    this_threshold = thresholds(ii);
    this_segmentation = segment_2d(depth, this_threshold, nms_width);
    
    % checking havent created too many segments
    total_segments = total_segments + length(unique(this_segmentation));
    if total_segments > max_segments
        break
    end
    
    idxs(ii, :) = this_segmentation;
end

% only return the unique segmentations
idxs(any(isnan(idxs), 2), :) = [];
idxs = unique(idxs, 'rows');
