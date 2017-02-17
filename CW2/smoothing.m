function [ beta ] = smoothing(pi, A, E, Y, C)

K = size(pi, 1);
T = size(Y, 2);

beta = zeros(K, T);
beta(:, T) = ones(K, 1); % init beta zT

for k = T-1:-1:1
    beta(:, k) = A*(beta(:, k+1).*E(:, Y(k+1))); % iterate
    beta(:, k) = beta(:, k)/C(k+1); % renormalise
end

end