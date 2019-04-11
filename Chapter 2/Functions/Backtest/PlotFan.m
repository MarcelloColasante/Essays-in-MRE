function h1 = PlotFan(time, X, color)
% This function display the fan plot of a sequence of processes
% reporting the median (solid line), the 50% percentile range (dim shading) 
% and the 90% percentile range (dimmer shading)

tmp = quantile(X, [0.05, 0.25, 0.5, 0.75, 0.95], 2);
h1 = fill([time; flipud(time)], [tmp(:, 1); flipud(tmp(:, 2))], color);
set(h1, 'FaceAlpha', 0.1, 'EdgeAlpha', 0.1);
hold on; h2 = fill([time; flipud(time)], [tmp(:, 2); flipud(tmp(:, 4))], color);
set(h2, 'EdgeColor', color, 'FaceAlpha', 0.3, 'EdgeAlpha', 0.3);
hold on; h3 = fill([time; flipud(time)], [tmp(:, 4); flipud(tmp(:, 5))], color);
set(h3, 'EdgeColor', color, 'FaceAlpha', 0.1, 'EdgeAlpha', 0.1);
plot(time, tmp(:, 3), 'LineWidth', 1.5, 'Color', color);

end