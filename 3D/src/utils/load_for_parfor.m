function out = load_for_parfor(filename, variablename)
% a workaround file for loading in a parfor loop

T = load(filename, variablename);
out = T.(variablename);

