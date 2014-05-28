function [binary, output_origin] = segments_to_binary(segments, outlier_idx)
% Converts matrix segmentation to binary segmentation
% segments is an N * M matrix, each row of which defines a different
% segmentation of the depth, into a number of different segments.
% THe output is a P * M binary matrix, where each row is a logicial array
% denoting one segment
% outlier_idx is an optional argument which specifies an index in segments
% which should be ignored for the purposes of forming the final binary
% matri
% origin is a vector showing which segmentation each final binary region
% came from

N = size(segments, 1);
M = size(segments, 2);

binary_cell = cell(N, 1);
origin_cell = cell(N, 1);

% loop over each segmentation
for ii = 1:N
    
    unique_idxs = nanunique(segments(ii, :));
    unique_idxs(isnan(unique_idxs)) = [];
    if nargin == 2
        unique_idxs(unique_idxs==outlier_idx) = [];
    end
    
    this_binary = zeros(length(unique_idxs), M);
    
    % extract each segment from this set of segments
    for jj = 1:length(unique_idxs)
    	this_binary(jj, :) = segments(ii, :) == unique_idxs(jj);
    end
    
    binary_cell{ii} = this_binary;
    origin_cell{ii} = ii * ones(size(this_binary, 1), 1);
    
end

binary = logical(cell2mat(binary_cell));
origin = cell2mat(origin_cell);

output_origin = zeros(size(binary, 1), N);
for ii = 1:length(origin)
    output_origin(ii, origin(ii)) = 1;
end
