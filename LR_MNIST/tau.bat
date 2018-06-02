rem "D:\INSA\CLANU\build-LR_MNIST-Imported_Kit_temporaire-Release\src\mnist_train_lrgd.exe" "D:\INSA\CLANU\LR_MNIST\data\\" "mnist_train.csv" "mnist_test.csv" "theta.csv" 200 0.6
@echo off
for %%t in (.3 .5 .7 .9 1.1 1.3 1.5) do (
%1 %2 %3 %4 %5 %6 %%t
)
pause
