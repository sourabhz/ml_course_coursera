function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%cost function
h_x = 1./(1 + exp(-(X*theta))); %this is hypothesis
first_term = (-y)'*(log(h_x));
second_term = (1-y)'*(log(1-h_x));
subtract = first_term - second_term;
subtract = subtract/m;
%regularized
theta(1) = 0;
regularised = (theta)'*theta;
regularised = (regularised*lambda)/(2*m);
J = subtract + regularised;
% =============================================================
%gradient
first = (h_x-y);
grad =  ((X)'*first)/m;
%regularized term for grad
reg = (lambda/m)*theta;
grad = grad + reg;
end
