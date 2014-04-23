function im_out = boxcrop_2d(im_in)
% crops zero-valued rows and columns from outside of an image

% getting sums of columns and rows
col_sums = any(im_in, 1);
row_sums = any(im_in, 2);

% finding start and end columns and rows
[~, top_col] = find(col_sums, 1, 'first');
[~, end_col] = find(col_sums, 1, 'last');

[~, top_row] = find(row_sums', 1, 'first');
[~, end_row] = find(row_sums', 1, 'last');

% cropping image
im_out = im_in(top_row:end_row, top_col:end_col);
