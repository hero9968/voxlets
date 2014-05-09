function plot3d ( XYZ, colour, sample_rate )
% PLOT3D - Plots the FARo data as a point cloud
%
% INPUTS
%   XYZ - Set of point cloud data in XYZ fomat. Acceptable arrays:
%               3 x (n*m)
%               (n*m) x 3
%               3 x n x m
%               n x m x 3

% TODO - Add automatic downsampling if point cloud is too big?? - maybe

% format input data

%{
if ndims(XYZ) == 2
    
    if size(XYZ, 2) == 3 && size(XYZ, 1) ~= 3
        XYZ = XYZ';
    elseif size(XYZ, 1) ~= 3
        error('One of the dimensions must be equal to 3!');
    end

elseif ndims(XYZ) == 3
    
    if size(XYZ, 1) == 3
        %fine
    elseif size(XYZ, 3) == 3
        XYZ = permute(XYZ, [3, 1, 2]);
    else
        error('First or last dimension must be 3');
    end
end
%}
XYZ = XYZ';

if nargin < 2
    colour = 'b';
end

if nargin < 3
  sample_rate = 1;
end

if size(colour, 2) == 3
  colour = colour';
end

XYZ = XYZ(:, :);

n_points = size(XYZ, 2);
idx = 1:sample_rate:n_points;
    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%if color is string or colour is no integer and 3 long
if isa(colour, 'char') || numel(colour) == 3

    X = XYZ(1, idx);
    Y = XYZ(2, idx);
    Z = XYZ(3, idx);
    %disp('plotting...');
    plot3(X, Y, Z, '.', 'MarkerSize', 10, 'MarkerFaceColor', colour, 'MarkerEdgeColor', colour);
    
% else if colour is a class label
elseif numel(colour) == numel(XYZ)/3
    colour = colour(:);
    hold_mem = ishold;
    a = nanunique(colour);
    %C = jet(length(a));
    C = jet(length(a));
    %C = [0.6 0.6 0.6; 1 0 0 ];
    for i = 1:length(a)
        %if any(XYZ(3, :)>0)
        %    keyboard
        %end
        plot3d(XYZ(:, colour==a(i)), C(i, :));
        hold on
    end
    
    % turning hold off if it was before
    if ~hold_mem
        hold off
    end
    
    
% else colour is stupid
elseif all(size(colour)==size(XYZ))
  
  % now plotting each point with a different colour...
  X = XYZ(1, idx);
  Y = XYZ(2, idx);
  Z = XYZ(3, idx);
  colour = colour(:, idx)';
  colour = double(colour) / 255;
  scatter3(X, Y, Z, 20, colour, 'filled');
  
else
  size(colour)
  size(XYZ)
    error('C must be a single color or a vector the same length as XYZ');
end
    
daspect([1 1 1])
xlabel('x');
ylabel('y');
zlabel('z');

end