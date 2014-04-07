function [output_image, output_image_cropped, transformed] = ...
    aggregate_depth_predictions(prediction, image_size)


assert(isfield(prediction, 'mask'));
assert(isfield(prediction, 'transform'));
assert(isfield(prediction, 'weight'));



% input checks

















