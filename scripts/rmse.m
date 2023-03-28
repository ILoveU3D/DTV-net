function value = rmse(x, y)
%calculate rMse(x,y)
mse = sum((x-y).^2);
value = 100 * mse / sum(y.^2);
end

