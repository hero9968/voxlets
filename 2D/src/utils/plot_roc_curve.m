function plot_roc_curve(TPR, FPR, colour, thresh)

assert(length(TPR) == length(FPR))

if nargin == 2 || isempty(colour)
    colour = 'b';
end

old_hold = ishold;

plot(TPR, FPR, colour); 
hold on

% plotting the dot
if nargin > 3
    [~, tidx] = min(abs(thresh-0.5));
    %H = plot(TPR(tidx), FPR(tidx), '.', 'markersize', 10, 'color', colour);
    H = plot(TPR(tidx), FPR(tidx), [colour(1), '.'], 'markersize', 10);

    % exclude point from the legend
    set(get(get(H,'Annotation'),'LegendInformation'),...
        'IconDisplayStyle','off');
    
end

if old_hold
    hold on
else
    hold off
end

axis equal; 
set(gca, 'xlim', [0, 1]), 
set(gca, 'ylim', [0, 1]); 
xlabel('FPR')
ylabel('TPR')
