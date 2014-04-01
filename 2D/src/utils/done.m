function done( number_done, total, skip, message )
% simple function to display how many have been completed of a total
%
% USAGE
%  done( number_done, total, [skip_number], [message] )
%
% CONTEXTUAL EXAMPLE
%  for ii = 1:100
%  	foo(ii);
%  	done( ii, 100, 5, 'my loop' )
%  end

% see if skipping
if nargin >= 3 && ~isempty(skip) && mod(number_done, skip) ~= 0
    return
end

if nargin < 4
    message = '';
end

% construct basic message
msg = ['Done ' num2str(number_done)];

if nargin > 1
  msg = [msg, ' of ' num2str(total) ' ' message];
end

% showing message
disp(msg);

