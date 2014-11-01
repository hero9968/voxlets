function plot_angles(angles_im)
% this function plots the angles im, nicely...

h = imagesc(angles_im);
set(h, 'alphadata', ~isnan(angles_im))
axis image
colorbar
hold on

[Y, X] = find(~isnan(angles_im));
alpha = 5;
to_use = randperm(length(X), 2000);

for ii = to_use
    
    
    this_angle = angles_im(Y(ii), X(ii));
    
    
    start_x = X(ii);% + cos(this_angle) * alpha;
    end_x = X(ii) + cos(this_angle) * alpha;
    start_y = Y(ii);% + sin(this_angle) * alpha;
    end_y = Y(ii) + sin(this_angle) * alpha;
    
    plot([start_x, end_x], [start_y, end_y], '-b')
    
end

hold off
