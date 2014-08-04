function V = full_3d(size, ind, vals)
% turns a 3d sparse representation into a full 3d matrix

assert(length(size)==3);
assert(isvector(ind))

if nargin == 2

    % creating logical array only
    V = false(size);
    V(ind) = true;

elseif nargin == 3

    % creating double array with the specified values in the specified locations
    assert(length(vals)==length(ind));
    V = zeros(size);
    V(ind) = vals;

end

