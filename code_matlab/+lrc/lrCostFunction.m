function [J, grad] = lrCostFunction(phi, X, y)

    %-- LRCOSTFUNCTION Compute cost and gradient for logistic regression
    %--   J = LRCOSTFUNCTION(phi, X, y, lambda) computes the cost of using
    %--   theta as the parameter for logistic regression and the
    %--   gradient of the cost w.r.t. to the parameters. 

    
    %-- Initialization of energy value J
    %J = 0;

    %-- Initialization of gradient vector
    [m,~] = size(X);
    %grad = zeros(1,n);
    
    
    % ====================== YOUR CODE HERE =========================
    % YOU SHOULD COMPUTE 
    %   - THE VALUE OF THE ENERGY FUNCTION J
    %   - THE GRADIENT VECTOR OF THE ENERGY FUNCTION J
    % ===============================================================
    
    
    c=1;

    h = lrc.sigmoid(X * phi');
    yc = y==c;
    
    J=-sum(yc.*log(h) + (1-yc).*log(1-h))/m;
    grad=sum(X .* (h-yc))/m;

end
