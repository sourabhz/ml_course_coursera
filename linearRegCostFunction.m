function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%
%regularised cost function
h_x = X*theta;
J_x = (h_x-y)'*(h_x-y);
J_x = (1/(2*m))*J_x;
theta(1)=0;
reg_x = ((lambda/(2*m))*(theta'*theta));
J = J_x + reg_x;


grad = (h_x-y)/m;
grad = X'*grad;
reg_grad = (lambda/m)*theta;
grad = grad+reg_grad;







% =========================================================================

grad = grad(:);

end
