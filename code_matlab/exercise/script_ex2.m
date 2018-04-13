clear all;
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

%-- Initialization of gradient vector
grad = zeros(1,n);


% ====================== YOUR CODE HERE =========================
% YOU SHOULD COMPUTE THE GRADIENT VECTOR OF THE ENERGY FUNCTION J
% ===============================================================


y_c = @(y, i, c) (y(i) == c);

theta = zeros(10, n);
for c=0:9
    for j=1:n
        h_c = @(x_i) lrc.sigmoid(x_i * transpose(theta(c+1, :)));
        theta(c+1, j) = grad_J(c, X, j, y, h_c, y_c);
    end
end

disp(grad*grad')
