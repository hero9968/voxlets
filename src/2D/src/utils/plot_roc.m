function [AUC, TPR, FPR, thresh ] = plot_roc(GT, pred, colour)
% my customised roc plotting

assert(all(size(GT)==size(pred)))

if nargin == 2
    colour = 'r';
end

%[TPR, FPR, thresh] = roc(GT(:)', pred(:)');
[TPR, FPR, thresh, AUC] = perfcurve2(GT(:)', pred(:)', 1);

if nargout == 0
    plot_roc_curve(TPR, FPR, colour, thresh);
end