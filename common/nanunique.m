function out = nanunique( in )
% An incomplete wrapper for unique, which removes repeating nan values.
% It's incomplete because it doesn't deal with arguments 2 and 3 from unique,
% or accept options like 'rows'
% (better versions are available on FEX)

in (isnan(in)) = [];
out = unique( in );

