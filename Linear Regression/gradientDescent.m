function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters,

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
    sub0=0;sub1=0;
    prediction0=0;prediction1=0;
    for i=1:m,
          prediction0=prediction0+(theta(1)*X(i,1)+theta(2)*X(i,2)-y(i));
          prediction1=prediction1+(theta(1)*X(i,1)+theta(2)*X(i,2)-y(i))*X(i,2);
    endfor
    sub0=prediction0*(alpha/m);
    sub1=prediction1*(alpha/m);
    theta(1)=theta(1)-sub0;
    theta(2)=theta(2)-sub1;


    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

endfor

end
