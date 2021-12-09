function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%


vals=[0.01;0.03;0.1;0.3;1;3;10;30];
op_c=0.01;
op_sigma=0.01;
n=length(vals);

model= svmTrain(X, y, op_c, @(x1, x2) gaussianKernel(x1, x2, op_sigma));
predictions=svmPredict(model,Xval);
op_error=mean(double(predictions ~= yval));


for i=1:n,
  test_c=vals(i);
  for j=1:n,
    test_sigma=vals(j);
    model= svmTrain(X, y, test_c, @(x1, x2) gaussianKernel(x1, x2, test_sigma));
    predictions=svmPredict(model,Xval);
    test_error=mean(double(predictions ~= yval));
    if test_error<op_error,
      op_error=test_error;
      op_c=test_c;
      op_sigma=test_sigma;
    endif
  endfor
endfor

C=op_c;
sigma=op_sigma;


% =========================================================================

end
