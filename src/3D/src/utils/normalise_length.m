function v_out = normalise_length(v_in)

v_out = v_in ./ repmat(sqrt(sum(v_in.^2, 2)), 1, size(v_in, 2));