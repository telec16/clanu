@echo off
for %%t in (.1 .3 .5 .7 .9 1 1.1) do (
%1 %2 %3 %4 %%t
)
pause
