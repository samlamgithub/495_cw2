function [ alpha, C ] = discrete_filtering(T, K, pi, A, E, Y)
% Out size:K x T, T x 1                   In size: K x 1, k x k, k x 6, 1 x T

C = zeros(T, 1); % scaling factors
alpha = zeros(K, T);  % init alpha
alpha(:, 1) = pi.* E(:, Y(1)); % init alpha z1
C(1) = sum(alpha(:,1));
alpha(:, 1) = alpha(:,1)/C(1); % normalise to get posterior

for t = 2:T
    alpha(:, t) = E(:, Y(t)).*(A'*alpha(:,t-1)); % calculate alpha values
    C(t) = sum(alpha(:,t));
    alpha(:,t) = alpha(:,t)./C(t); % renormalise, divided by sclaing factor
    error = 0.01;
    assert(abs(sum(alpha(:,t))-1.000)< error); % asserting normalise is correct
end

end