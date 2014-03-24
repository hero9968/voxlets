% script to generate subset of the MPEG-7 dataset
% ... and to save a struct with the subset info

params

%% 
all_files = dir(paths.mpeg);
to_remove = ismember({all_files.name}, {'.', '..', '.DS_Store'});
all_files(to_remove) = [];

%% find all the class names
class_name = cell(1, length(all_files));

for ii = 1:length(all_files)
    if strcmp(all_files(ii).name(end-3:end), '.gif')
        t_name = all_files(ii).name(1:end-4);
        split_name = strsplit(t_name, '-');
        class_name{ii} = split_name{1};
    else
        class_name{ii} = '';
    end
end

[unique_classes, ~, class_idx] = unique(class_name);
unique_classes(1) = [];

%% now for each unique class, copy the first 3 examples to the new foler
old_path = [paths.mpeg, '%s-%d.gif'];
system(['rm  ' paths.subset '/*'])
filelist = [];

for ii = 1:length(unique_classes)
    inliers = cellfun(@(x)(strcmp(x, unique_classes{ii})), class_name);
    this_class_idx = find(inliers, 3, 'first');
    
    for jj = 1:3
        this_name = all_files(this_class_idx(jj)).name;
        this_path = [mpeg_path, this_name];
        copyfile(this_path, subset_path);
        
        % maintaining a list of the subset info
        this_struct.name = this_name;
        this_struct.class = unique_classes{ii};
        this_struct.class_idx = ii;
        
        filelist = [filelist, this_struct];
        
    end   

end

save(paths.subset_files, 'filelist')