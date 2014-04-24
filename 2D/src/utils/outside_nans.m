function [idx, start_nans, end_nans] = outside_nans(input)
% returns a logical vector indicating the location of the nans on the
% outside of the input vector, i.e. from the start to the first non-nan and
% from the last non-nan until the end.

% ensuring row vector
input = input(:)';

width = length(input);

% finding nan locations
first_nonnan = findfirst(~isnan(input)', 1, 1, 'first');
last_nonnan = findfirst(~isnan(input)', 1, 1, 'last');

% creating logical array
idx_nums = [1:(first_nonnan-1), (last_nonnan+1):width];
idx = false(size(input));
idx(idx_nums) = true;

start_nans = first_nonnan - 1;
end_nans = width - last_nonnan;
