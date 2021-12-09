function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

for i=1:m,
  cost=y(i)*log(sigmoid(theta'*X(i,:)'))+(1-y(i))*log(1-sigmoid(theta'*X(i,:)'));
  J=J+cost;
endfor
J=J*(-1/m);

reg=0;
for i=2:length(theta),
  reg=reg+theta(i)^2;
endfor
reg=reg*(lambda/(2*m));
J=J+reg;

for i=1:size(X,2),
  prediction=0;
  for j=1:m,
    prediction=prediction+(sigmoid(theta'*X(j,:)')-y(j))*X(j,i);
  endfor
  if i==1,
    grad(i)=prediction/(m);
  else
    grad(i)=(prediction/(m))+(lambda/m)*theta(i);
  endif
  
endfor




% =============================================================

end
