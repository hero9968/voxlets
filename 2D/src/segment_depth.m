% a script (to become a function) to segment a depth image into separate
% parts, ideally creating some kind of soup of segments
clear
cd ~/projects/shape_sharing/2D/src

addpath utils
%%
load('~/projects/shape_sharing/data/2D_shapes/raytraced/171_27_mask.mat', 'this_raytraced_depth')
XY = [1:length(this_raytraced_depth); this_raytraced_depth];

%%

[norms, curve] = normals_radius_2d( XY, 5);
curve

tXY = (XY(2, :));
first_deriv = diff(tXY);
second_deriv = diff(first_deriv);

clf
subplot(211)
plot_normals_2d(XY, norms)

subplot(212)

%axis image

%plot(XY(1, :), XY(2, :)/100)
hold on
plot(XY(1, :), abs([first_deriv, 0]), 'r')
plot(XY(1, :), [second_deriv, 0, 0], 'g')
plot(XY(1, :), curve*10, 'c')
hold off
%axis image

%% dividing via the first derivative...
threshold = 10;
derv = abs([first_deriv, 0]);
[~, idx] = sort(derv, 2, 'descend');

%%
plot(derv);
derv_sup = non_max_sup_1d(derv, 3, 0);
hold on
plot(non_max_sup_1d(derv, 3, 0),'r')
hold off

threshold = 10;

% sort derivates and remove those out of range...
[sorted, idx] = sort(derv_sup(:), 1, 'descend');
to_remove = sorted < threshold;
sorted(to_remove) = [];
idx(to_remove) = [];

% generate a hierachical segmentation
split_points = zeros(1, size(XY, 2));
split_points(idx+1) = 1
final_segments = cumsum(split_points)
scatter(XY(1, :), XY(2, :), 10, final_segments);

% find the hierarchy and store in a cell array as a soup of segments
% we don't need to be too explicit about storing the hierachy we just need 
% the possible groups of segments
% begin with all segments and indivudal segments
segs = unique(final_segments);
soup = [{segs}, num2cell(segs)];

%%
derv_sup(derv_sup<threshold) = 0;
segments = cumsum(derv_sup);
plot(segments)

%%
while 1
     
    % 
    soup = [soup, {20}]
    
    break
    
end
soup


%% lets find out how many segments each of the depth images
% is segmented into
cd ~/projects/shape_sharing/2D/src
run ../define_params
load(paths.split_path, 'split')
load(paths.test_data)
addpath external
%
%close all

%%
threshold = 10;
curve_threshold = 5;
number_segments = nan(1, length(split.test_data));

% loop over each test image
for ii = 75%:length(split.test_data)


    % loading in the depth for this image
    this_filename = split.test_data{ii};
    this_depth_path = fullfile(paths.raytraced, this_filename);
    load([this_depth_path '.mat'], 'this_raytraced_depth');
    
    

    first_deriv = diff(this_raytraced_depth);
    derv = abs([first_deriv, 0]);
    derv_sup = non_max_sup_1d(derv, 3, 0)';
    
    %second_deriv = (diff(derv));
    %second_deriv_sup = [0; non_max_sup_1d(second_deriv, 3, 0)]';
        
    %L = length(this_raytraced_depth);
    %XY = [1:L; this_raytraced_depth];
    %[norms, curve] = normals_radius_2d( XY, 7);
    
    % taking the dot product between adjacient pairs of normals
    %norm_diff = ones(1, length(norms));
    %for jj = 1:(size(norms, 2)-1)
    %     norm_diff(jj) = dot(norms(:, jj), norms(:, jj+1));
    %end
    %curve = 1 - norm_diff;

    % sort derivates and remove those out of range...
    %[sorted, idx] = sort(derv_sup(:), 1, 'descend');
    %to_remove = sorted < threshold;
    %sorted(to_remove) = [];
    %idx(to_remove) = [];

    split_points = [derv_sup >= threshold];% | second_deriv_sup >= curve_threshold];
    
    %differences = derv_sup(find(split_points)-1);
    
    % generate a hierachical segmentation
    %
    % = zeros(1, L);
    %split_points(idx+1) = 1;
    split_points = circshift(split_points, [0, 1]);
    sum(split_points);
    final_segments = cumsum(split_points);
    
    unique(final_segments)
    number_segments(ii) = length(unique(final_segments));
    subplot(211)
    scatter(1:length(this_raytraced_depth), this_raytraced_depth, 10, final_segments);
    subplot(212)
    %plot(curve);
    %hold     off
    %axis image
    
    drawnow
    %pause(0.5);
   ii 
end
colormap(jet)

%clf

%%
figure
bar(accumarray(number_segments', 1))

%%
thresholds = 5:5:40;
segments = nan(length(thresholds), length(this_raytraced_depth));
for ii = 1:length(thresholds)
    t = thresholds(ii);
    this_segmentation = segment_2d(this_raytraced_depth, t, 3);
    segments(ii, :) = this_segmentation;
    ii
end
final_segments = unique(segments, 'rows')


%%
for ii = 1:num_segments
    for jj = 1:num_segments
                
    end
end






