close all
h= figure()
plot(mean(medioid.tpr), mean(medioid.fpr), 'b')
hold on
plot(mean(modal.tpr), mean(modal.fpr), 'r--')
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