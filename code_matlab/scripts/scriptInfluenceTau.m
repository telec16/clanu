clear;
close all;
clc;
addpath('../');


%-- parameters
fromTau = 0.01; %-- gradient descent: start number of learning rate coefficient 
stepsTau = 10;  %-- gradient descent: step number of learning rate coefficient
toTau = 2.5;    %-- gradient descent: end number of learning rate coefficient
epsilon = 0.01;    %-- gradient descent: convergence thresholder
maxIter = 40;   %-- gradient descent: iterations
methods = ["simple", "conjugate"];
 

%-- mnist database location
url = 'https://www.creatis.insa-lyon.fr/~bernard/ge/';
local_data_path = '../data/';
local_param_path = '../param/';


%-- Downlad minst database
filename_db = 'mnist.mat';
if (~exist([local_data_path,filename_db],'file'))
     tools.download(filename_db,url,local_data_path);
end

%-- Load mnist database
load([local_data_path,filename_db]);
widthDigit = size(training.images,2);
heightDigit = size(training.images,1);


num_labels = 10;          %-- 10 labels, from 0 to 9



%-- Create X matrix
X = zeros(size(test.images,3),widthDigit*heightDigit+1);
for k=1:size(test.images,3)
    digit = test.images(:,:,k);
    X(k,:) = [1,digit(:)'];
end


%-- Create y vector
y = test.labels;
m = size(X,1);


%--  train logistic regression method
figure;
hold on;
accuracies = struct;
for method = methods
    fprintf('\n\n\nTraining Logistic Regression with %s gradient descent methode depending on learning rate coefficient...\n\n\n', method);

    accuracy=zeros(1, stepsTau);
    for i=1:stepsTau
        tau=fromTau+(i-1)*((toTau-fromTau)/(stepsTau-1));
        fprintf('\n\nCurrently: %4i\n\n', tau);
        
        [all_theta] = lrc.train(X, y, num_labels, maxIter, epsilon, tau, method);
        pred = lrc.predict(all_theta, X);
        accuracy(i) = mean(double(pred == y) * 100);
    end
    
    accuracies.(method) = accuracy;
    plot(accuracy);
end
hold off;

fprintf('\n\n\nEnd. Thank you.\n');
