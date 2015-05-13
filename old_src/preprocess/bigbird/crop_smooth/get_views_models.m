function [views, models] = get_views_models()

base_path = get_base_path();

% get the directories
fid = fopen([base_path, 'bigbird/bb_to_use.txt']);
models = textscan(fid, '%s\n');
models = models{1};
fclose(fid);

% get the views
fid = fopen([base_path, 'bigbird/poses_to_use.txt']);
views = textscan(fid, '%s\n');
views = views{1};
fclose(fid);