function [u, v] = best_subplot_dims( n_plots )
% function to find the best subplot dims for a number of plots to plot
% want u and v to be similar, but not waste too much space at the bottom

% special cases
switch n_plots
  case 3
    u = 1; v = 3;
  case 7
    u = 2; v = 4;
  case 8
    u = 2; v = 4;
  case 15
    u = 3; v = 5;
  otherwise
    v = ceil(sqrt(n_plots));
    u = ceil(n_plots/v);  
end
