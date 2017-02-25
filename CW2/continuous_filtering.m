function [ alpha, C ] = continuous_filtering(T, K, pi, A, E, Y)
% Out size:K x T, T x 1                   In size: K x 1, k x k, k x 6, 1 x T

% scaling factors
C = zeros(T, 1);
alpha = zeros(K, T); % init alpha
for k = 1:K
    alpha(k, 1) = pi(k)* normpdf(Y(1),E.mu(k), sqrt(E.sigma2(k))); % init alpha z1
end

C(1) = sum(alpha(:,1));
alpha(:, 1) = alpha(:,1)/C(1); % normalise to get posterior

for t = 2:T
    Emis_prob = zeros(K,1);
    for k = 1:K
       Emis_prob(k) = normpdf(Y(t), E.mu(k), sqrt(E.sigma2(k)));
    end
    alpha(:, t) = Emis_prob.*(A'*alpha(:,t-1)); % calculate alpha values
    C(t) = sum(alpha(:,t));
    alpha(:,t) = alpha(:,t)./C(t); % renormalise, divided by sclaing factor
    error = 0.01;
    assert(abs(sum(alpha(:,t))-1.000)<error); % asserting normalise is correct
end

end