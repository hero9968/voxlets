function [final_image, M] = soft_max(M, dim, alpha, weights)
% similar functionality to e.g. matlab's SUM or MEAN, but does noisy OR

%assert(all(M(:)>=0))
%assert(all(M(:)<=1))

N = size(M, dim);

if ndims(M) > 3
    error('Not sure if this works for dimensions above three...')
end

% applying weights
if nargin == 4
    assert(N == length(weights));
    
    if dim == 1
        M = bsxfun(@times, M, weights(:));
    elseif dim == 2
        for ii = 1:N
            M(:, ii, :) = M(:, ii, :) * weights(ii); 
        end
    elseif dim == 3
        for ii = 1:N
            M(:, :, ii, :) = M(:, :, ii, :) * weights(ii); 
        end
    else
        error('Seem to be asking for dimenion that does not exist')
    end
end

% forming final result
alpha_M = alpha * M;

% removing a constant offset to avoid overflow
% see e.g. http://lingpipe-blog.com/2009/03/17/softmax-without-overflow/
to_remove = alpha * max(M, [], dim);
if dim == 1
    alpha_M = alpha_M - repmat(to_remove, [size(M, 1), 1]);
elseif dim == 2
    alpha_M = alpha_M - repmat(to_remove, [1, size(M, 2)]);
elseif dim == 3
    alpha_M = alpha_M - repmat(to_remove, [1, 1, size(M, 3)]);
end

exp_b = exp(alpha_M);

numerator = sum(M .* exp_b, dim);
denominator = sum(exp_b, dim);

if any(~isfinite(numerator(:))) || any(~isfinite(denominator(:)))
    error('Infinity occured')
end

final_image = numerator ./ denominator;

