function dispInitialParams(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)

Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);

disp('')
%disp('nn_params size')
%disp(size(nn_params))
%disp('nn_params')
%disp(nn_params((1:3)))
%disp('...')
disp('input_layer_size')
disp(input_layer_size)
disp('hidden_layer_size')
disp(hidden_layer_size)
disp('K/num_labels')
disp(num_labels)
disp('Theta1 size')
disp(size(Theta1))
disp('Theta1')
disp(Theta1((1:3), (1:3)))
disp('...')
disp('Theta2 size')
disp(size(Theta2))
disp('Theta2')
disp(Theta2((1:3), (1:3)))
disp('...')
disp('m')
disp(m)
disp('X size')
disp(size(X))
disp('X')
disp(X((1:3), (1:3)))
disp('...')
disp('y size')
disp(size(y))
disp('y')
disp(y((1:3),:))
disp('...')
disp('lambda')
disp(lambda)
disp('')

end