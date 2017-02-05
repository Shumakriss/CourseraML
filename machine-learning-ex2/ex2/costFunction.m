function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

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
%
% Note: grad should have the same dimensions as theta
%

%Cost
z=sum((theta' .* X)')';
hOfX=sigmoid(z);

negativeY=-1*y;
logOfHOfX=log(hOfX);
leftTerm=negativeY.*logOfHOfX;

oneMinusY=1-y;
logOfOneMinusHOfX=log(1-hOfX);
rightTerm=oneMinusY.*logOfOneMinusHOfX;

difference = leftTerm - rightTerm;
summation = sum(difference);
J=summation/m;

%Gradient
hOfXMinusY = hOfX .- y;
hOfXMinusYTimesXj = hOfXMinusY .* X;
grad = (sum(hOfXMinusYTimesXj) ./ m)';



% =============================================================

end
