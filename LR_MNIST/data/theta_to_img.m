clear; close all;
theta = csvread('theta.csv');
theta = theta(:,1:(end-1));

cols=4;
imgW=28;

lines=(ceil(10/cols));
thetaN = zeros(imgW*lines, imgW*cols);
for line=1:lines
    temp = zeros(imgW, imgW*cols);
    for col=1:cols
        x = (line-1)*cols+col;
        if x<=10
            img = splitLines(theta(x,:),imgW);
            temp(:, ((col-1)*imgW+1):col*imgW) = img;
        end
    end
    thetaN(((line-1)*imgW+1):line*imgW , :) = temp;
end

thetaG = mat2gray(thetaN);

thetaC = zeros([size(thetaG) 3]);
thetaC(:,:,1) = thetaG;
thetaC(:,:,2) = 1-thetaG;

figure;
imshow(thetaC);
figure;
imshow(thetaG);

function img = splitLines(v, c)
    l=length(v) / c;
    img = zeros(l,c);
    for i=1:l
        img(i,:)=v(((i-1)*c+1):i*c);
    end
end

