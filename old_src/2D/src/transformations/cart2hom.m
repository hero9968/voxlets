function hom = cart2hom(cart)
% converts 2d cartesian coords to homogeneous

assert(size(cart, 1) == 2);

hom = [cart; ones(1, size(cart, 2))];