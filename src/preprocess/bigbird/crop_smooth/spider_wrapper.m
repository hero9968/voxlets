function sp = spider_wrapper(xyz, norms, edges, focal_length)

% creating input stack
xyz_reshaped = reshape(xyz, [size(edges), 3]);
norms_reshaped = reshape(norms, [size(edges), 3]);

in_stack = cat(3, edges, xyz_reshaped, norms_reshaped);

% running spider mex file
sp = spider_angled(in_stack);

% converting nan points to nan
for ii = 1:size(sp, 3)
    t = sp(:, :, ii);
    t(isnan(xyz(:, 1))) = nan;
    sp(:, :, ii) = t;
end

% normalising pixel distances by the depth and focal length
depth = xyz_reshaped(:, :, 3);
for ii = [ 1, 4, 7, 10]
   sp(:, :, ii) =  (sp(:, :, ii) .* depth) / focal_length;
end
