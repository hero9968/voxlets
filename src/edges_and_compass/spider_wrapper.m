function sp = spider_wrapper(xyz, norms, edges, mask, focal_length)

% creating input stack
xyz_reshaped = reshape(xyz, [size(edges), 3]);
norms_reshaped = reshape(norms, [size(edges), 3]);

in_stack = double(cat(3, edges, xyz_reshaped, norms_reshaped));

% running spider mex file
sp1 = spider(in_stack);
sp2 = spider_angled(in_stack);


sp = cat(3, sp1, sp2);

% converting nan points to nan
for ii = 1:size(sp, 3)
    t = sp(:, :, ii);
    t(mask==0) = nan;
    sp(:, :, ii) = t;
end

% normalising pixel distances by the depth and focal length
depth = xyz_reshaped(:, :, 3);
for ii = [ 1, 4, 7, 10, 13, 16, 19, 22]
   sp(:, :, ii) =  (sp(:, :, ii) .* depth) / focal_length;
end

