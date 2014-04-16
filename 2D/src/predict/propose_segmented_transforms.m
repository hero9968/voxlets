function transforms = propose_segmented_transforms(model, depth, binary_segments, params)
% wrapper for propose transforms which uses the segmented image to propose
% transforms for each segment in turn.

%params.num_proposals = 5;
%params.apply_known_mask = 0;
%params.transform_type = 'pca';

min_segment_size = 5; % number of points required to bother trying to fit

% convert the segmentation to binary format, so each row is a segment
%binary_segments = segments_to_binary(segments);

transforms = cell(1, size(binary_segments, 1));

% choose how many proposals to make for each of the segments
num_pixels_per_segment = sum(binary_segments, 2);
to_remove = num_pixels_per_segment < min_segment_size;
num_pixels_per_segment(to_remove) = 0;
pixel_distribution = num_pixels_per_segment / sum(num_pixels_per_segment);
num_proposals_per_segment = round(params.num_proposals * pixel_distribution);

% loop over every segment
for ii = 1:size(binary_segments, 1)
    
    % removing depths outside this segment
    this_segment_idx = binary_segments(ii, :);
    this_depth = depth;
    this_depth(~this_segment_idx) = nan;

    % propose transforms just for this segment
    these_params = params;
    these_params.num_proposals = num_proposals_per_segment(ii);
    if these_params.num_proposals > 0
        transforms{ii} = propose_transforms(model, this_depth, these_params);
    end
    
    %plot_transforms(transformed, out_img_cropped, gt_image);
    
end

transforms(cellfun(@isempty, transforms)) = [];
transforms = cell2mat(transforms);