function check_isgood_transform( trans )

if any(isnan(trans(:))) 
    disp('Seems like the transform is not very nice')
    keyboard
end

if cond(trans') > 1e8
    disp('Seems like conditioning is bad')
    keyboard
end
