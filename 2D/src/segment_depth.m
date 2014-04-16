% a script (to become a function) to segment a depth image into separate
% parts, ideally creating some kind of soup of segments
clear
close all
cd ~/projects/shape_sharing/2D/src
run ../define_params
load(paths.split_path, 'split')
load(paths.test_data)
addpath external
addpath utils
addpath segment

%%
plotting = 1;
threshold = 10;
curve_threshold = 5;
number_items_in_soup = nan(1, length(split.test_data));
segmentation = cell(1, length(split.test_data));

% loop over each test image
for ii = 11%:length(split.test_data)


    % loading in the depth for this image
    this_filename = split.test_data{ii};
    this_depth_path = fullfile(paths.raytraced, this_filename);
    load([this_depth_path '.mat'], 'this_raytraced_depth');
    
    % doing the segmentation
    segmentation{ii} = segment_soup_2d(this_raytraced_depth, params.segment_soup);
    %segmentation{ii} = segmentation{ii}(:, 1:params.im_width);
    
    %bin = filter_segments(segmentation{ii});
    
    if plotting
        subplot(211)
        plot(1:length(this_raytraced_depth), this_raytraced_depth, 'o');
      %  axis image
        %subplot(312)
        %plot(segmentation{ii}');
        
        subplot(212)
        %bin= segments_to_binary(segmentation{ii});
        imagesc(segmentation{ii})

        drawnow
        %pause(0.5);
    end    
    
    % count the number of items in the soup
    number_items_in_soup(ii) = size(segmentation{ii}, 1);
    
    ii 
end


%% plot graph of number of items
clf
bar(accumarray(number_items_in_soup', 1))
xlabel('Number of items in soup')
ylabel('Frequency')

%% plot graph of number of segments in each soup item
all_segments = cell2mat(segmentation');
num_segments = max(all_segments, [], 2)+1;

clf
bar(accumarray(num_segments, 1))
xlabel('Total number of segments')
ylabel('Frequency')

%% plot graph of total number of segments for each depth image
total_segments = @(X)sum(max(X, [], 2)+1);
T = cellfun(total_segments, segmentation);
clf
bar(accumarray(T', 1))
xlabel('Number of items in soup')
ylabel('Frequency')


%rowfun(@unique, all_segments)




