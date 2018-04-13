clear;
close all;
clc;
addpath('../');


%-- parameters
maxIter = 100;   %-- maximum number of iterations


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


%-- Perform training
num_labels = 10;          %-- 10 labels, from 0 to 9


%-- Create X matrix
X = zeros(size(training.images,3),widthDigit*heightDigit+1);
for k=1:size(training.images,3)
    digit = training.images(:,:,k);
    X(k,:) = [1,digit(:)'];
end


%-- Create y vector
y = training.labels;
[m,n] = size(X);


%-- Load pre-learned parameters
filename_param = 'param_ex1_2.mat';
load([local_param_path,filename_param]);


%-- Initialization of energy value J
J = 0;


% ====================== YOUR CODE HERE =========================
% YOU SHOULD COMPUTE THE VALUE OF THE ENERGY FUNCTION J
% ===============================================================


%Ok, let's try this the hard&ugly way
for c=0:9
    J=0;
    for i=1:m
        if(y(i) == c)
            yc = 1;
        else
            yc = 0;
        end

        h = lrc.sigmoid(X(i, :) * phi');

        J = J + yc*log(h) + (1-yc)*log(1-h);
    end
    disp(-J/m)
end


% for k=1:3
%     figure
%     imshow(mat2gray( training.images(:,:,k)));
% end

% apparently, c=0 (tested with t = @(i) [h_c(phi, X, i), y(i)];)
c=0;

y_c = @(y, i, c) (y(i) == c);
h_c = @(theta_c, X, i) lrc.sigmoid(X(i, :) * transpose(theta_c));
J_c_i = @(theta_c, X, y, c, i) ...
         y_c(y, i, c)  * log(    h_c(theta_c, X, i)) + ...
    (1 - y_c(y, i, c)) * log(1 - h_c(theta_c, X, i));
J_c = @(theta_c, X, y, c) ...
    (-1/m) * tools.sigma(1, m, @(i) J_c_i(theta_c, X, y, c, i));

J=0;
for i=1:m
    J = J + J_c_i(phi, X, y, c, i);
end
J=-J/m;

disp(J_c(phi, X, y, c))
disp(J)

%The issue is clearly not from my function. I must have misunderstood
%something at some point...
disp(J_c_i(phi, X, y, 0, 1))
disp(log(1 - lrc.sigmoid(X(1, :) * phi')))

disp(J_c_i(phi, X, y, 0, 2))
disp(log(    lrc.sigmoid(X(2, :) * phi')))

