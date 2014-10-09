function bb = load_bigbird(modelname, view)

[~, name] = system('hostname');
name = strtrim(name);

if strcmp(name, 'michaels-mbp.lan') || strcmp(name, 'mfirman.cs.ucl.ac.uk')
    base_path = '/Users/Michael/projects/shape_sharing/data/';
elseif strcmp(name, 'troll')
    base_path = '/mnt/scratch/mfirman/data/';
else
    error('No host found')
end

obj_path = [base_path, 'bigbird/', modelname];
%depth_path = [base_path,

bb.depth = h5read([obj_path, '/' , view, '.h5'], '/depth')';
bb.depth = double(bb.depth) / 10000;
bb.mask = ~imread([obj_path, '/masks/' , view, '_mask.pbm']);
bb.rgb =  imread([obj_path, '/' , view, '.jpg']);

temp = strsplit(view, '_');
bb.cam_name = temp{1};

% loading the intrinsics
bb.K_rgb = h5read([obj_path, '/calibration.h5'], ['/' bb.cam_name '_rgb_K'])';
bb.K_depth = h5read([obj_path, '/calibration.h5'], ['/' bb.cam_name '_depth_K'])';

% loading the extrinsics
bb.H_rgb = h5read([obj_path, '/calibration.h5'], ['/H_' bb.cam_name '_from_NP5']);
bb.H_ir = h5read([obj_path, '/calibration.h5'], ['/H_' bb.cam_name '_ir_from_NP5']);
%bb.H_mesh = h5read([obj_path, '/calibration.h5'], ['/H_' bb.cam_name '_ir_from_NP5']);