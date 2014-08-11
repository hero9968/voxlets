classdef pcloud < handle
% PCLASS is a point cloud class
% stores a cloud and associated depth image, normals, etc
% this is for a *structured* cloud, where each point is associated with a point
% in a depth image. I will create another class for unstructured points...
    
    properties
        
        % NAME      SIZE                % DESCRIPTION
        depth       % H x W             depth image
        rgb         % H x W x 3         colour image
        mask        % H x W             location of non-nan points in image
        xyz         % (H x W) x 3       3d points
        normals     % (H x W) x 3       normals of the 3d points
        curvature   % (H x W) x 1       curvature of the 3d points
        viewpoint   % 4 x 4             viewpoint and viewing direction
        segmentsoup % (H x W) x num_segments   ...
                    %                   binary array assigning points to segments
        intrinsics  % 3 x 3             intrinsic matrix
        plane_rotate % 4 x 4            matrix to rotate points to align with scene's dominant plane
        
    end
    
    methods (Access = public)
        
        function obj = pcloud(varargin)
            
        % loading in a cloud
        
            disp(num2str(nargin))
            
            if nargin == 1 && ischar(varargin{1})
                
                [~, ~, ext] = fileparts(varargin{1});
                
                disp(ext)
                
                switch ext
                    case '.pgm'
                        
                        obj.depth = readpgm(varargin{1});
                        obj.set_as_kinect();
                        obj.xyz = projection(obj.depth, obj.intrinsics);
                        
                        % inserting nans
                        obj.mask = obj.depth > 0;
                        obj.depth(~obj.mask) = nan;
                        obj.xyz(~obj.mask, :) = nan;
                        
                        obj.sanity_check();
                        
                    case '.pcd'
                        % not entirely sorted out yet...
                        P = loadpcd(varargin{1});
                        
                        obj.xyz = P(:, :, 1:3);
                        obj.xyz = reshape(permute(obj.xyz, [3, 1, 2]), 3, [])';
                        obj.depth = P(:, :, 3);
                        
                        if size(P, 2) > 3
                            obj.rgb = P(:, :, 4:6);
                        end
                end
            end
            
        end
        
        function showdepth(obj)    
            imagesc(obj.depth)
            axis image
        end
        
        function plot3d(obj, varargin)
            plot3d(obj.xyz_non_nan, varargin{:}) 
        end
        
        function obj = project_depth(obj, intrinsics)
        % project the depth image into the 3d points
        
            if nargin == 2
                obj.intrinsics = intrinsics;
            end
            
            obj.xyz = projection(obj.depth, obj.intrinsics);

            obj.sanity_check();
            
        end
        
        function obj = set_as_kinect(obj)
        % sets the defaults for kinect (v1) images
        
            obj.intrinsics = [];
            
            focal_length = 240/(tand(43/2));
            obj.intrinsics = [focal_length, 0, 320; ...
                              0, focal_length, 240; ...
                              0, 0, 1];           
        end
        
        function sanity_check(obj)
        % checking all the objects are of the correct sizes
        
            assert(isequal(size(obj.depth), size(obj.mask)))

            %if ~isempty(obj.rgb)
            %    assert(size(obj.rgb, 1) == obj.height)
            %    assert(size(obj.rgb, 2) == obj.width)
            %end
            height = size(obj.mask, 1);
            width = size(obj.mask, 2);
            
            assert(size(obj.xyz, 1) == height*width);
            assert(size(obj.xyz, 2) == 3);
            
            if ~isempty(obj.normals)
                assert(size(obj.normals, 1) ==  height*width)
            end
            
            % todo - checking the nan locations are all in the same place
        end
        
        function segment = extract_segment(obj, idx)
        % extracting a segment from an index number
        
            assert(~isempty(obj.segmentsoup), 'Must have a segmentation to extract a segment')
            assert(size(obj.segmentsoup, 1) == size(obj.xyz, 1), ...
                'Segmentation must be same size as point cloud')
        
            if strcmp(idx, 'all')
                idx = 1:size(obj.segmentsoup, 2);
            end
                       
            for ii = 1:length(idx)
                this_segment = obj.segmentsoup(:, idx(ii));
                segment(ii) = extract_segment_from_indices(obj, this_segment);
            end
        
        end
        
        function segment = extract_segment_from_indices(obj, indices)    
        % extracts a segment from the cloud based on a vector of indices
        % indices can either be logical array or array of index values
            
            segment.xyz = obj.xyz(indices, :);
            segment.normals = obj.normals(indices, :);
            segment.viewpoint = obj.viewpoint;
            
            % extracting the full mask
            segment.mask = 0 * obj.mask;
            segment.mask(indices) = true;
            
        end
        
        function xyz = xyz_non_nan(obj, idx)
        % get just the non-nan xyz
        
            xyz = obj.xyz(obj.mask(:), :);
            
            if nargin == 2
                xyz = xyz(idx, :);
            end
           
        end
        
        function normals = normals_non_nan(obj, idx)
        % get just the non-nan normals
        
            normals = obj.normals(obj.mask(:), :);
            
            if nargin == 2
                normals = normals(idx, :);
            end
           
        end
        
        function curvature = curvature_non_nan(obj, idx)
        % get just the non-nan curvature
        
            curvature = obj.curvature(obj.mask(:));
            
            if nargin == 2
                curvature = curvature(idx, :);
            end
           
        end
        
        
        function cloud_size = get_size(obj, dim)
        % getting a size of the depth image
            
            cloud_size = size(obj.depth);
            
            if nargin == 2
                cloud_size = cloud_size(dim);
            end
        end
        
        
    end
    
    methods (Access = private)

    end
end



% Utility functions

function xyz = projection(depth, intrinsics)
% project a depth image into 3d using specified intrinsics

    assert(isequal(size(intrinsics), [3, 3]))

    im_height = size(depth, 1);
    im_width = size(depth, 2);

    % stack of homogeneous coordinates of each image cell
    [xgrid, ygrid] = meshgrid(1:im_width, 1:im_height);
    full_stack = [xgrid(:) .* depth(:), ygrid(:).* depth(:), depth(:)];

    % apply inverse intrinsics, and convert to standard coloum format
    xyz = (intrinsics \ full_stack')';

end