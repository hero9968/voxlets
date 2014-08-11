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
        idx         % 1                 number segment in the original cloud

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
            
        end
        
        
        
        
        
        
    end
    
    
    
    
end