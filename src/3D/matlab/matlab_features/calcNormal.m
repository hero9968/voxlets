function [ normal, spread, condition_number, d ] = calcNormal( N, p )
%CALCNORMAL Calculates normal of point p, based on reletive distances of
% surrounding points N.

% Inputs:
%       N       xyz coords of point to consider and surrounders
%       p       xyz coords of point to consider

% OUTPUTS:
%       normal  3 x 1 direction of normal at point
%       spread  spread over the points i.e. the size of the mininmum
%       eigenvalue
% d is basically lalonde features



% if ( nargin ~= 1 )
%     error('Wrong number of input arguments!');
% end
% 
% if size(N, 2) ~= 3
%     error('N must be a n x 3 matrix')
% end
% 
% if isempty(N)
%     error('N must not be empty');
% end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%finding normal

%remove nans
N(any(isnan(N), 2), :) = [];

if size(N, 1) >= 3 && size(N, 2) == 3

    % comining neighbours with centre point
    [A, D] = eig(cov(N));

    % finding minimus eigenvalue
    d = diag(D);
    ind = d == min(d);
    ind_max = d == max(d);
    col = find( ind );
    col_max = find( ind_max );
    spread = d(col(1));
    
    % condition number - bigger is more stable, smaller is less stable
    condition_number =  d( col_max(1) ) / spread;
    
    % extracting corresponding eigenvector
    normal = A(:, col(1))';
    
    % rotating to make normal point towards [0 0 0]
    if nargin == 2 
      t_p = p ./ sqrt(sum(p.^2));
      t_normal = normal ./ sqrt(sum(normal.^2));
      cos_angle = dot( t_p, t_normal );
      if cos_angle < 0
        normal = -normal;
      end
    end
%	end
    
	d = flipud(d);
	
else
    normal = [0 0 0];
	spread = 0;
end

