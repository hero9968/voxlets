function idxs = segment_soup_2d(depth, params)
% segments a 2d depth measurement into a variety of different sements, each
% of which is represented as a row in a matrix

%thresholds = 5:5:40;

% setup and input checks
assert(isvector(depth));

thresholds = params.segment_soup.thresholds;
nms_width = params.segment_soup.nms_width;

% form matrix to be filled
idxs = nan(length(thresholds), length(depth));

% segment for each different distance threshold
for ii = 1:length(thresholds)
    this_threshold = thresholds(ii);
    idxs(ii, :) = segment_2d(depth, this_threshold, nms_width);
end

% only return the unique segmentations
idxs = unique(idxs, 'rows');
