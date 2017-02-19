function [ beta ] = discrete_smoothing(pi, A, E, Y, C)
%                         k x 1, k x k, k x 6, 1 x T, T x 1
K = size(pi, 1);
T = size(Y, 2);

beta = zeros(K, T);
beta(:, T) = ones(K, 1); % init beta zT

for t = T-1:-1:1
    % K x 1     k x k   k x 1     k  x 1
    beta(:, t) = A*(beta(:, t+1).*E(:, Y(t+1))); % iterate
    % k x 1      k x 1       1 x 1
    beta(:, t) = beta(:, t)/C(t+1); % renormalise
end

end