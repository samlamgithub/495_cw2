function [ beta ] = continuous_smoothing(T, K, A, E, Y, C)
% Out size: K x T                 In size:  K x 1, k x k, k x 2, 1 x T, T x 1

beta = zeros(K, T); % init beta
beta(:, T) = ones(K, 1); % init beta zT
for t = T-1:-1:1
    Emis_prob = zeros(K,1); % init Emission Probability
    for k = 1:K
       Emis_prob(k) = normpdf(Y(t+1), E.mu(k), sqrt(E.sigma2(k)));
    end
    beta(:, t) = A*(beta(:, t+1).*Emis_prob); % calculate beta values
    % k x 1      k x 1       1 x 1
    beta(:, t) = beta(:, t)/C(t+1); % renormalise, divided by scaling factor
end

end

