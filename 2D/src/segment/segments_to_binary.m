function binary = segments_to_binary(segments)
% Converts matrix segmentation to binary segmentation
% segments is an N * M matrix, each row of which defines a different
% segmentation of the depth, into a number of different segments.
% THe output is a P * M binary matrix, where each row is a logicial array
% denoting one segment

N = size(segments, 1);
M = size(segments, 2);

binary_cell = cell(N, 1);

for ii = 1:N
    
    unique_idxs = unique(segments(ii, :));
    this_binary = zeros(length(unique_idxs), M);
    
    % extract each segment from this set of segments
    for jj = 1:length(unique_idxs)
    	this_binary(jj, :) = segments(ii, :) == unique_idxs(jj);
    end
    
    binary_cell{ii} = this_binary;
    
end

binary = logical(cell2mat(binary_cell));