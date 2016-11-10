function v = sysMult(x, B, w, mode)

if(mode ==1)
    v = w.*(B*x);
else
    v = B'*(w.*x);
end