basep = '~/projects/shape_sharing/data/bigbird_cropped';
outp = '~/projects/shape_sharing/media/supplementary/imgs/bigbird/rgb_ims/';


names = {'ritz_crackers', 'sunkist_fruit_snacks_mixed_fruit',...
    'vo5_extra_body_volumizing_shampoo', ...
    'red_bull', ...
    'quaker_big_chewy_chocolate_chip', ...
    'pringles_bbq', ...
    'pringles_bbq', ...
    'spongebob_squarepants_fruit_snaks'}

    
views = {'NP3_0',    'NP4_0','NP2_216','NP4_0','NP1_0','NP1_312','NP4_0','NP2_0'}

for ii = 1:length(names)
   full = fullfile(basep, names{ii},  [views{ii}, '.mat'])
   A = load(full)
   outpath = [outp, '/', names{ii}, '_', views{ii}, '.png']
   imwrite(A.rgb, outpath)
   
    
end

%%

clear views

names = {'tapatio_hot_sauce', 'nutrigrain_harvest_blueberry_bliss', 'progresso_new_england_clam_chowder', 'pringles_bbq'}
    views = {'NP1_312', 'NP1_0', 'NP2_216', 'NP1_312'};
    
outp = '~/Desktop/extra/'

for ii = 1:length(names)
   full = fullfile(basep, names{ii},  [views{ii}, '.mat'])
   A = load(full)
   outpath = [outp, '/', names{ii}, '_', views{ii}, '.png']
   imwrite(A.rgb, outpath)
   outpath2 = [outp, '/', names{ii}, '_', views{ii}, '_depth.png']
   imwrite(convert_to_jet(A.depth, min(A.depth(:)), max(A.depth(:))), outpath2)
end
