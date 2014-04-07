function out = noisy_or(M, dim, weights)
% similar functionality to e.g. matlab's SUM or MEAN, but does noisy OR

assert(all(M(:)>=0))
assert(all(M(:)<=1))

N = size(M, dim);

if ndims(M) > 3
    error('Not sure if this works for dimensions above three...')
end

% applying weights
if nargin == 3
    assert(N == length(weights));
    
    if dim == 1
        for ii = 1:N
            M(ii, :, :) = M(ii, :, :) * weights(ii); 
        end
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
out = 1 - prod(1-M, dim);
