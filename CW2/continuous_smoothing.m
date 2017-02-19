function [ beta ] = continuous_smoothing(pi, A, E, Y, C)
%                         k x 1, k x k, k x 2, 1 x T, T x 1
K = size(pi, 1);
T = size(Y, 2);

beta = zeros(K, T);
beta(:, T) = ones(K, 1); % init beta zT

for t = T-1:-1:1
    Emis_prob = zeros(K,1);
    for k = 1:K
       Emis_prob(k) = normpdf(Y(t+1), E.mu(k), E.sigma2(k));
    end
    % K x 1     k x k   k x 1     k  x 1
    beta(:, t) = A*(beta(:, t+1).*Emis_prob); % iterate
    % k x 1      k x 1       1 x 1
    beta(:, t) = beta(:, t)/C(t+1); % renormalise
end

end

