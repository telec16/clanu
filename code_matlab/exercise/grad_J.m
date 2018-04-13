function g = grad_J(theta_c, X, y, c, j)
%GRAD_J Compute the gradient of J_j, the energy
%   theta_c is a [1 x n] vector which has been learned
%   X is a [m x n] matrix that store each pictures
%   y is a [m x 1] vector that contains the value of the m picture
%   c is the current digit
%   j is a scalar that correspond to a pixel in a picture


%   y_c(y, i, c) is a function that test weither or not y(i) equals c
y_c = @(y, i, c) (y(i) == c);

%   h_c(x) is a function that evaluate the probability that the value of x is c
h_c = @(theta_c, X, i) lrc.sigmoid(X(i, :) * transpose(theta_c));

%   x is a [m x 1] vector that contains the pixel j of the m pictures
x = X(:, j); 
m = length(x);

g = (1/m) * tools.sigma(1, m, @(i) x(i) * (h_c(theta_c, X, i) - y_c(y, i, c)));

end
