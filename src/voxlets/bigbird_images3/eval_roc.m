close all
halfway_point = length(medioid.tpr)/2

h = figure()

tpr = mean(medioid.tpr);
fpr = mean(medioid.fpr);
plot(tpr, fpr, 'b')
hold on
plot(tpr(halfway_point), fpr(halfway_point), 'bo')
tpr = mean(modal.tpr);
fpr = mean(modal.fpr);
plot(tpr, fpr, 'r--')
plot(tpr(halfway_point), fpr(halfway_point), 'ro')
hold off

axis equal
xlim([0, 1])
ylim([0, 1])
t1= xlabel('TPR');
t2 = ylabel('FPR');

legend({'Medioids of tree votes', 'Modal tree votes'}, 'Location', 'SouthEast')

set(gca, 'FontSize', 18)
set(t1, 'FontSize', 18)
set(t2, 'FontSize', 18)

saveas(gca, './data/roc_curve.eps', 'epsc2')