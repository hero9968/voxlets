basep = '~/projects/shape_sharing/data/bigbird_cropped'
outp = '~/projects/shape_sharing/media/report/imgs/data/rgb_ims/'


names = {'tapatio_hot_sauce',          'zilla_night_black_heat',          'nutrigrain_harvest_blueberry_bliss',          'pop_tarts_strawberry'}
views = {'NP1_312', 'NP2_312', 'NP1_0', 'NP3_0'}

for ii = 1:length(names)
   full = fullfile(basep, names{ii},  [views{ii}, '.mat'])
   A = load(full)
   outpath = [outp, '/', names{ii}, '.png']
   imwrite(A.rgb, outpath)
   
    
end