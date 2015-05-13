function cart = hom2cart(hom)
% converts 2d homogeneous coords to cartesian 

assert(size(hom, 1) == 3);

X = hom(1, :);
Y = hom(2, :);
div = hom(3, :);

cart = [X./div; Y./div];