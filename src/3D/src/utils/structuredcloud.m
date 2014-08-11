classdef structuredcloud < handle
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
        
        function self = structuredcloud(varargin)
            
        % loading in a cloud
        
            if nargin == 1 && ischar(varargin{1})
                
                [~, ~, ext] = fileparts(varargin{1});
                
                switch ext
                    case '.pgm'
                        
                        self.depth = readpgm(varargin{1});
                        self.set_as_kinect();
                        self.xyz = projection(self.depth, self.intrinsics);
                        
                        % inserting nans
                        self.mask = self.depth > 0;
                        self.depth(~self.mask) = nan;
                        self.xyz(~self.mask, :) = nan;
                        
                        self.sanity_check();
                        
                    case '.pcd'
                        % not entirely sorted out yet...
                        P = loadpcd(varargin{1});
                        
                        self.xyz = P(:, :, 1:3);
                        self.xyz = reshape(permute(self.xyz, [3, 1, 2]), 3, [])';
                        self.depth = P(:, :, 3);
                        
                        if size(P, 2) > 3
                            self.rgb = P(:, :, 4:6);
                        end
                end
            end
            
        end
        
        function showdepth(self)    
            imagesc(self.depth)
            axis image
        end
        
        function plot3d(self, varargin)
            plot3d(self.xyz_non_nan, varargin{:}) 
        end
        
        function self = project_depth(self, intrinsics)
        % project the depth image into the 3d points
        
            if nargin == 2
                self.intrinsics = intrinsics;
            end
            
            self.xyz = projection(self.depth, self.intrinsics);

            self.sanity_check();
            
        end
        
        function self = set_as_kinect(self)
        % sets the defaults for kinect (v1) images
        
            self.intrinsics = [];
            
            focal_length = 240/(tand(43/2));
            self.intrinsics = [focal_length, 0, 320; ...
                              0, focal_length, 240; ...
                              0, 0, 1];           
        end
        
        function sanity_check(self)
        % checking all the objects are of the correct sizes
        
            assert(isequal(size(self.depth), size(self.mask)))

            %if ~isempty(self.rgb)
            %    assert(size(self.rgb, 1) == self.height)
            %    assert(size(self.rgb, 2) == self.width)
            %end
            height = size(self.mask, 1);
            width = size(self.mask, 2);
            
            assert(size(self.xyz, 1) == height*width);
            assert(size(self.xyz, 2) == 3);
            
            if ~isempty(self.normals)
                assert(size(self.normals, 1) ==  height*width)
            end
            
            % todo - checking the nan locations are all in the same place
        end
        
        function segment = extract_segment(self, idx)
        % extracting a segment from an index number
        
            assert(~isempty(self.segmentsoup), 'Must have a segmentation to extract a segment')
            assert(size(self.segmentsoup, 1) == size(self.xyz, 1), ...
                'Segmentation must be same size as point cloud')
        
            if strcmp(idx, 'all')
                idx = 1:size(self.segmentsoup, 2);
            end
                       
            for ii = 1:length(idx)
                this_segment = self.segmentsoup(:, idx(ii));
                segment(ii) = extract_segment_from_indices(self, this_segment);
            end
        
        end
        
        function seg = extract_segment_from_indices(self, indices)    
        % extracts a segment from the cloud based on a vector of indices
        % indices can either be logical array or array of index values
            
            seg = segment;
            seg.xyz = self.xyz(indices, :);
            seg.normals = self.normals(indices, :);
            seg.curvature = self.curvature(indices);
            seg.viewpoint = self.viewpoint;
            seg.plane_rotate = self.plane_rotate;
            
            % extracting the full mask
            seg.mask = reshape(indices, size(self.mask));
            
        end
        
        function xyz = xyz_non_nan(self, idx)
        % get just the non-nan xyz
        
            xyz = self.xyz(self.mask(:), :);
            
            if nargin == 2
                xyz = xyz(idx, :);
            end
           
        end
        
        function normals = normals_non_nan(self, idx)
        % get just the non-nan normals
        
            normals = self.normals(self.mask(:), :);
            
            if nargin == 2
                normals = normals(idx, :);
            end
           
        end
        
        function curvature = curvature_non_nan(self, idx)
        % get just the non-nan curvature
        
            curvature = self.curvature(self.mask(:));
            
            if nargin == 2
                curvature = curvature(idx, :);
            end
           
        end
        
        
        function cloud_size = get_size(self, dim)
        % getting a size of the depth image
            
            cloud_size = size(self.depth);
            
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