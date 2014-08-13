classdef segment < handle
% SEGMENT stores the geometry etc of a segment of a depth image
    

    properties
        
        % NAME      SIZE                % DESCRIPTION
        rgb         % N x 3             colour of points
        mask        % H x W             location where the points came from in original image
        xyz         % N x 3             3d points
        normals     % N x 3             normals of the 3d points
        curvature   % N x 1             curvature of the 3d points
        viewpoint   % 4 x 4             viewpoint and viewing direction of original cloud
        intrinsics  % 3 x 3             intrinsic matrix of original cloud
        plane_rotate % 4 x 4            matrix to rotate points to align with scene's dominant plane
        scale       % 1                 size of the segment, as measured...
        idx         % 1                 number segment in the original cloud
        features    % 1                 structure holding all the required features
                    %               (keeping as a structure for now for flexibility)
        transform_to_origin  
                    % 4x4               transform from 3D points to origin
        % NOTE: there are N points in the segment
        
    end
    
    methods (Access = public)
        
        function plot3d(self, varargin)
            plot3d(self.xyz, varargin{:}) 
        end
        
        function sanity_check(self)
        % checking all the object are of the correct sizes
        
            N = size(self.xyz, 1);
                    
            assert(size(self.normals, 1) == N);
            assert(size(self.xyz, 2) == 3);
            
            assert(length(self.curvature) == N);
            
            if ~isempty(self.normals)
                assert(size(self.normals, 1) ==  N)
            end
            
            assert(sum(sum(self.mask))==N)
            
        end
        
        function self = estimate_scale(self)
            self.scale = estimate_size(self.xyz);
        end
        
        function xyz = scaled_xyz(self)
            xyz = self.xyz / self.get_scale();
        end
        
        function self = compute_transform_to_origin(self)
            % estimates the transforms from the 3d points to the origin

            transforms.centroid = centroid(self.mask);
            
            centroid_linear_index = ...
                centroid_2d_to_linear_index(transforms.centroid, self.mask);

            transforms.centroid_3d.xyz = nanmedian(self.xyz, 1);
            transforms.centroid_3d.norm = self.normals(centroid_linear_index, :);

            % compute the centroid normal - not just from the point but from a few more also...
            [~, neighbour_idx] = ...
                pdist2(self.xyz, transforms.centroid_3d.xyz, 'euclidean', 'smallest', 1000);

            neighbour_xyz = self.xyz(neighbour_idx, :);
            transforms.centroid_normal = calcNormal( neighbour_xyz, transforms.centroid_3d.xyz );

            % FINAL TRANSFORMATION MATRIX

            % translation from the origin to the scene segment
            trans2 = translation_matrix_3d(transforms.centroid_3d.xyz);
            rot2 = inv(transformation_matrix_from_vector(transforms.centroid_normal, 1));
            scale_segment = scale_matrix_3d(self.get_scale());

            % combining
            transforms.final_M = trans2 * rot2 * scale_segment; 
            
            self.transform_to_origin = transforms.final_M;
            
            
        end
        
        function scale = get_scale(self)
            
            if isempty(self.scale)
                self.estimate_scale();
            end
            
            scale = self.scale;
            
        end
        
        function self = load_from_depth(self, depth, intrinsics)
            % function to set the xyz points and mask from a depth image
            
            self.mask = ~isnan(depth);
            self.intrinsics = intrinsics;
            
            self.xyz = projection(depth, intrinsics);
            self.xyz(~self.mask, :) = [];
                        
        end
        
        
        
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