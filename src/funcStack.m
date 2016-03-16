function f = funcStack(x, f1,f2, m)
f = [f1(x(1:m,:)); f2(x(m+1:end,:))];
end