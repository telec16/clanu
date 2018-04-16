function [all_theta] = train(X, y, num_labels, maxIter, epsilon, tau, approach) 

    %-- [all_theta] = train(X, y, num_labels, maxIter, epsilon, tau)
    %-- Trains multiple logistic regression classifiers and returns all
    %-- the classifiers in a matrix all_theta, where the i-th row of all_theta 
    %-- corresponds to the classifier for label i
    %-- approach can be "conjugate" or "simple" (default)

    if (~exist('approach','var'))
        approach = "simple";
    end
    
    %-- Get the number of samples m and the size of the parameter vector
    [~,n] = size(X);

    %-- Loop over each class
    all_theta = zeros(num_labels,n);
    for c=1:num_labels

        %-- Compute theta for class c

        %-- Set Initial theta
        initial_theta = zeros(1,n);
        
        %-- Run gradient descent method to update theta values
        options = struct('MaxIter',maxIter,'epsilon',epsilon,'tau',tau);
        if(approach == "conjugate")
            [theta] = lrc.conjugate_gradient( ...
            @(t)(lrc.lrCostFunction(t, X, (y == c-1))), ...
            initial_theta, options);
        else
            [theta] = lrc.gradient_descent( ...
            @(t)(lrc.lrCostFunction(t, X, (y == c-1))), ...
            initial_theta, options);
        end
        
        %-- Store corresponding result into all_theta matrix
        all_theta(c,:) = theta;

    end

end
