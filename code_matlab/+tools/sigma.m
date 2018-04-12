%sigma(1, 10, @(i) i)
function sig = sigma(s, e, fct)

sig=0;
for i=s:e
    sig=sig+ fct(i);
end

end