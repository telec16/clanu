clear all;
close all;
clc;
addpath('../');

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

%-- Create X matrix for train
X = zeros(size(training.images,3),widthDigit*heightDigit+1);
for k=1:size(training.images,3)
    digit = training.images(:,:,k);
    X(k,:) = [1,digit(:)'];
end


%-- Create y vector for train
y = training.labels;
m = size(X,1);

%-- Create X matrix for test
Xt = zeros(size(test.images,3),widthDigit*heightDigit+1);
for k=1:size(test.images,3)
    digit = test.images(:,:,k);
    Xt(k,:) = [1,digit(:)'];
end
yt = test.labels;
m = size(Xt,1);

%Number of labels
num_labels = 10; 

%--  train logistic regression method -> gradient descent
disp('\nTraining Logistic Regression with simple gradient descent methode depending on iteration number...\n')
tau=1;
epsilon=0;
methode = 0;
accuracy_gradient_descent=zeros(1,10);
for i=1:10
    maxIter=20+(i-1)*20;
    [all_theta] = lrc.train(X, y, num_labels, maxIter, epsilon, tau, methode);
    pred = lrc.predict(all_theta, Xt);
    accuracy_gradient_descent(i)= mean(double(pred == yt) * 100);
end

%--  train logistic regression method -> conjugate gradient
disp('\nTraining Logistic Regression with conjugate gradient depending on iteration number...\n')
tau=1;
epsilon=0;
methode = 1;
accuracy_conjugate_gradient=zeros(1,10);
for i=1:10
    maxIter=20+(i-1)*20;
    [all_theta] = lrc.train(X, y, num_labels, maxIter, epsilon, tau, methode);
    pred = lrc.predict(all_theta, Xt);
    accuracy_conjugate_gradient(i)= mean(double(pred == yt) * 100);
end

figure;
plot(20:18:200,accuracy_gradient_descent,'c');
hold on;
plot(20:18:200,accuracy_conjugate_gradient,'b');

