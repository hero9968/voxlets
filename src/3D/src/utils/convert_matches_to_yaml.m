function matches = convert_matches_to_yaml(all_matches)
% helper function to convert the all_matches structure to a format
% well-suited for writing to a yaml file.
% Only writes out some of the fields

unique_objects = unique({all_matches.object_name});
matches = {};

for ii = 1:length(unique_objects)    
    this_matches = find(ismember({all_matches.object_name}, unique_objects{ii}));
    match.name = unique_objects{ii};
    match.transform = {};

    for jj = 1:length(this_matches)
        transM = all_matches(this_matches(jj)).vox_transformation;
        transform.T = transM(1:3, 4)';
        transform.R = transM(1:3, 1:3);% * [0, 1, 0; -1, 0, 0; 0, 0, 1];
        transform.weight = 1;
        transform.region = all_matches(this_matches(jj)).region;
        match.transform{end+1} = transform;
    end
    matches{end+1} = match;
end