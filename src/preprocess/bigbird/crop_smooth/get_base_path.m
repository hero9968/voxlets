function base_path = get_base_path()

[~, name] = system('hostname');
name = strtrim(name);

if strcmp(name, 'troll')
    base_path = '/mnt/scratch/mfirman/data/';
else
    base_path = '/Users/Michael/projects/shape_sharing/data/';
end
