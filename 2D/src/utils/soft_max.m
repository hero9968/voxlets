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
        %for ii = 1:N
        %    M(ii, :, :) = M(ii, :, :) * weights(ii); 
        %end
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
exp_b = exp(alpha * M);

numerator = sum(M .* exp_b, 3);

denominator = sum(exp_b, 3);

final_image = numerator ./ denominator;
final_image(isnan(final_image)) = 0;


