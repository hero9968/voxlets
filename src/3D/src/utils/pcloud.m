classdef pcloud < handle
% PCLASS is a point cloud class
% stores a cloud and associated depth image, normals, etc
    
    properties
        
        depth       % height x width depth image
        rgb         % height x width x 3 colour image
        xyz         % 3d points without any nans
        normals     % normals of the 3d points
        curvature   % curvature of the 3d points
        mask        % location of the non-nan points in image space
        viewpoint   % 4x4 matrix specifying viewpoint and viewing direction
        segmentsoup % num_points x num_segments binary array giving assignment of points to segments
        intrinsics  % 3x3 intrinsic matrix
        plane_rotate % 4x4 transformation matrix to rotate the points to align with the scene's dominant plane
        
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
                        nan_locations = obj.depth==0;
                        obj.xyz(nan_locations(:), :) = nan;
                        obj.depth(nan_locations) = nan;
                        obj.mask = ~isnan(obj.depth);
                        
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
        % showing the depth image nicely
            
            imagesc(obj.depth)
            axis image
        end
        
        
        function plot3d(obj, vargin)
        % plot the 3d points nicely
        
            plot3d(obj.xyz, vargin)
        end
        
        function obj = project_depth(obj, intrinsics)
        % project the depth image into the 3d points
        
            if nargin == 2
                obj.intrinsics = intrinsics;
            end
            
            obj.xyz = projection(obj.depth, obj.intrinsics);
            
            % removing points which fall outside the mask
            obj.xyz(~obj.mask(:), :) = [];
            
            obj.sanity_check;
            
        end
        
        function obj = set_as_kinect(obj)
        % sets the defaults for kinect (v1) images
        
            obj.intrinsics = [];
            
            focal_length = 240/(tand(43/2));
            obj.intrinsics = [focal_length, 0, 320; ...
                              0, focal_length, 240; ...
                              0, 0, 1];           
        end
        
        function obj = extract_mask(obj)
        % extracts the mask from the depth image
        
            obj.mask = ~isnan(obj.depth);
        end
        
        function sanity_check(obj)
        % checking all the objects are of the correct sizes
        
            assert(isequal(size(obj.depth), size(obj.mask)))

            %if ~isempty(obj.rgb)
            %    assert(size(obj.rgb, 1) == obj.height)
            %    assert(size(obj.rgb, 2) == obj.width)
            %end
            
            N = sum(sum(~isnan(obj.depth)));
            
            assert(size(obj.xyz, 1) == N);
            
            if ~isempty(obj.normals)
                assert(size(obj.normals, 1) ==  N)
            end
            
            assert( sum(sum(obj.mask)) == N);
                        
        end
        
        function segment = extract_segment(obj, idx)
        % extracting a segment from an index number
        
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
            t_idx = find(obj.mask);
            segment.mask = 0 * obj.mask;
            segment.mask(t_idx(indices)) = true;
            
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