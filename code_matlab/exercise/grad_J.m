function g = grad_J(c, X, j, y, h_c, y_c)
%GRAD_J Compute the gradient of J_j, the energy
%   c is the current digit
%   X is a [m x n] matrix that store each pictures
%   j is a scalar that correspond to a pixel in a picture
%   y is a [m x 1] vector that contains the value of the m picture
%   h_c(x) is a function that evaluate the probability that the value of x is c
%   y_c(y, i, c) is a function that test weither or not y(i) equals c

x = X(:, j); %   x is a [m x 1] vector that contains the pixel j of the m pictures
m = length(x);

g = (1/m) * tools.sigma(1, m, @(i) x(i) * (h_c(X(i, :)) - y_c(y, i, c)));

end
