names = ['tapatio_hot_sauce_NP1_312.mat_%s_view_%d', 
         'zilla_night_black_heat_NP2_312.mat_%s_view_%d', 
         'nutrigrain_harvest_blueberry_bliss_NP1_0.mat_%s_view_%d', 
         'pop_tarts_strawberry_NP3_0.mat_%s_view_%d', 
         'vo5_extra_body_volumizing_shampoo_NP2_216.mat_%s_view_%d',
         'spongebob_squarepants_fruit_snaks_NP2_0.mat_%s_view_%d', 
         'red_bull_NP4_0.mat_%s_view_%d',
         'pringles_bbq_NP3_0.mat_%s_view_%d']
            
names = ['ritz_crackers_NP3_0_%s_view_180',
    'sunkist_fruit_snacks_mixed_fruit_NP4_0_%s_view_90',
    'vo5_extra_body_volumizing_shampoo_NP2_216_%s_view_0',
    'red_bull_NP4_0_%s_view_270',
    'quaker_big_chewy_chocolate_chip_NP1_0_%s_view_0',
    'pringles_bbq_NP1_312_%s_view_180',
    'pringles_bbq_NP4_0_%s_view_0',
    'spongebob_squarepants_fruit_snaks_NP2_0_%s_view_90']
              #'pringles_bbq_NP3_0.mat_%s_view_%d']
#         'vo5_extra_body_volumizing_shampoo_NP2_216.mat_%s_view_%d',]
objnames = ['ritz_crackers', 'sunkist_fruit_snacks_mixed_fruit',
    'vo5_extra_body_volumizing_shampoo', 
    'red_bull', 
    'quaker_big_chewy_chocolate_chip', 
    'pringles_bbq', 
    'pringles_bbq', 
    'spongebob_squarepants_fruit_snaks']

views = [90, 180, 0, 90, 90, 180, 180, 180]
before = '\includegraphics[height=\\turnheight, clip=true, trim=60 30 30 5]{'
after = '.png} &'

types = ['visible_pixels', 'gt', 'bb', 'zheng', 'oma']
#'zheng'
for (view, name) in zip(objnames, names):
    # image by itself
    print before + view + after
    for ii, itype in enumerate(types):
        fname = name % (itype)#, view)
        if ii == len(types)-1:
            print before + fname + '} \\\\'
        else:
            print before + fname + after