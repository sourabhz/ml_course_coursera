function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
%J = 0;
%grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%
%cost function
h_x = 1./(1 + exp(-(X*theta))); %this is hypothesis
first_term = (-y)'*(log(h_x));
second_term = (1-y)'*(log(1-h_x));
subtract = first_term - second_term;
subtract = subtract/m;
%regularized
%theta(1) = 0;
%regularised = (theta)'*theta;
%regularised = (regularised*1)/(2*m);
J = subtract;

%Gradient
first = (h_x-y)/m;
grad = (X)'*first;

% =============================================================

end
