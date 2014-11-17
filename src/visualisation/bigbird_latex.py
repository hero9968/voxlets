names = ['tapatio_hot_sauce_NP1_312.mat_%s_view_%d', 
         'zilla_night_black_heat_NP2_312.mat_%s_view_%d', 
         'nutrigrain_harvest_blueberry_bliss_NP1_0.mat_%s_view_%d', 
         'pop_tarts_strawberry_NP3_0.mat_%s_view_%d', 
         'vo5_extra_body_volumizing_shampoo_NP2_216.mat_%s_view_%d',
         'spongebob_squarepants_fruit_snaks_NP2_0.mat_%s_view_%d', 
         'red_bull_NP4_0.mat_%s_view_%d',
         'pringles_bbq_NP3_0.mat_%s_view_%d']
            
names = ['tapatio_hot_sauce_NP1_312.mat_%s_view_%d',
         'zilla_night_black_heat_NP2_312.mat_%s_view_%d',
         'nutrigrain_harvest_blueberry_bliss_NP1_0.mat_%s_view_%d',
                  'pop_tarts_strawberry_NP3_0.mat_%s_view_%d']
#         'vo5_extra_body_volumizing_shampoo_NP2_216.mat_%s_view_%d',]
views = [90, 180, 0, 90, 90, 180, 180, 180]
before = '\includegraphics[width=0.32\columnwidth, clip=true, trim=30 30 30 60]{data/renders_turn_table/'
after = '} &'

types = ['visible_pixels', 'gt', 'bb', 'zheng', 'oma']

for (view, name) in zip(views, names):
    for ii, itype in enumerate(types):
        fname = name % (itype, view)
        if ii == len(types)-1:
            print before + fname + '} \\\\'
        else:
            print before + fname + after