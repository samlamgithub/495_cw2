function [ beta ] = discrete_smoothing(T, K, A, E, Y, C)
% Out size: K x T                 In size: k x k, k x 6, 1 x T, T x 1

beta = zeros(K, T); % init beta
beta(:, T) = ones(K, 1); % init beta zT

for t = T-1:-1:1
    beta(:, t) = A*(beta(:, t+1).*E(:, Y(t+1))); %  % calculate beta values
    beta(:, t) = beta(:, t)/C(t+1); % renormalise, divided by scaling factor
end

end