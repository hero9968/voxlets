function [AUC, TPR, FPR ] = plot_roc(GT, pred, colour)
% my customised roc plotting

assert(all(size(GT)==size(pred)))

if nargin == 2
    colour = 'r';
end

%[TPR, FPR, thresh] = roc(GT(:)', pred(:)');
[TPR, FPR, thresh, AUC] = perfcurve2(GT(:)', pred(:)', 1);

[~, tidx] = min(abs(thresh-0.5));

if nargout > 0
    old_hold = ishold;

    plot(TPR, FPR, colour); 
    hold on
    %H = plot(TPR(tidx), FPR(tidx), '.', 'markersize', 10, 'color', colour);
    H = plot(TPR(tidx), FPR(tidx), [colour(1), '.'], 'markersize', 10);

    % exclude point from the legend
    set(get(get(H,'Annotation'),'LegendInformation'),...
        'IconDisplayStyle','off');

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
end