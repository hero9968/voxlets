D = load('/media/michael/Seagate/raw_nyu/nyu_depth_v2_labeled.mat')


for ii = 1:size(D.rawRgbFilenames)
    if D.rawRgbFilenames{ii} == '1000_bedroom_0057_3'
        disp ii
    end
end