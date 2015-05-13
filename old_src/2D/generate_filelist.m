% script to get list of all the files in the shape dataset, and store information
% like their name, class, idx etc.

clear
define_params

%%
all_files = dir(paths.mpeg);
to_remove = ismember({all_files.name}, {'.', '..', '.DS_Store'});
all_files(to_remove) = [];

%% setting up the structure with a name and class for each 
for ii = 1:length(all_files)
    
    [~, this_name, this_extension] = fileparts(all_files(ii).name);
    
    if strcmp(this_extension, '.gif')
        
        split_name = strsplit(this_name, '-');
        filelist(ii).class = split_name{1};
        filelist(ii).subclass = str2double(split_name{2});
        filelist(ii).name = this_name;
        filelist(ii).filename = all_files(ii).name;
        
    end
end

%% finding unique classes and their idxs
[unique_classes, ~, class_idx] = unique({filelist.class});

%% applying class idxs
for ii = 1:length(filelist)
    filelist(ii).class_idx = class_idx(ii);    
end

%% saving filelist to disk
save(paths.filelist, 'filelist')