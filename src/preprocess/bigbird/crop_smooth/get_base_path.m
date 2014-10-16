function base_path = get_base_path()

[~, name] = system('hostname');
name = strtrim(name);

if strcmp(name, 'michaels-mbp.lan') || strcmp(name, 'mfirman.cs.ucl.ac.uk')
    base_path = '/Users/Michael/projects/shape_sharing/data/';
elseif strcmp(name, 'troll')
    base_path = '/mnt/scratch/mfirman/data/';
else
    error('No host found')
end
