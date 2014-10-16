function bb = load_bigbird(modelname, view)

base_path = get_base_path();
obj_path = [base_path, 'bigbird/', modelname];

% loading the images from bigbird
bb.depth = h5read([obj_path, '/' , view, '.h5'], '/depth')';
bb.depth = single(bb.depth) / 10000;
bb.mask = ~imread([obj_path, '/masks/' , view, '_mask.pbm']);
bb.rgb =  imread([obj_path, '/' , view, '.jpg']);

temp = strsplit(view, '_');
bb.cam_name = temp{1};

% loading the intrinsics
bb.K_rgb = h5read([obj_path, '/calibration.h5'], ['/' bb.cam_name '_rgb_K'])';
bb.K_depth = h5read([obj_path, '/calibration.h5'], ['/' bb.cam_name '_depth_K'])';

% loading the extrinsics - this doesn't need to be scale
bb.H_rgb = h5read([obj_path, '/calibration.h5'], ['/H_' bb.cam_name '_from_NP5']);
bb.H_ir = h5read([obj_path, '/calibration.h5'], ['/H_' bb.cam_name '_ir_from_NP5']);

% loading the front render and the back render
temp = load([base_path, '/bigbird_renders/', modelname, '/', view, '_renders.mat'], 'front', 'back');
bb.front_render = single(temp.front);
bb.back_render = single(temp.back);

% doing all the scaling here
bb.scale_factor = 0.5; % this is the linear scale factor of how much smaller the RGB image will be

bb.K_rgb_original = bb.K_rgb;
bb.K_rgb(1, 1) = bb.K_rgb(1, 1) * bb.scale_factor;
bb.K_rgb(2, 2) = bb.K_rgb(2, 2) * bb.scale_factor;
bb.K_rgb(1, 3) = bb.K_rgb(1, 3) * bb.scale_factor;
bb.K_rgb(2, 3) = bb.K_rgb(2, 3) * bb.scale_factor;

bb.rgb = imresize(bb.rgb, bb.scale_factor);
bb.mask = imresize(bb.mask, bb.scale_factor);

%bb
assert(size(bb.rgb, 2)==size(bb.front_render, 2))
assert(size(bb.rgb, 2)==size(bb.back_render, 2))
assert(size(bb.rgb, 1)==size(bb.mask, 1))
