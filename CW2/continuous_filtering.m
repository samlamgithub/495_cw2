function [ alpha, C ] = continuous_filtering(pi, A, E, Y)
%                             k x 1, k x k, k x 2, 1 x T

% p(x_t|z_t) = product k=1 to 3: normal(x_t| mu(k), sigma(k)) ^ z_tk
% normpdf(X,mu,sigma)
K = size(pi, 1);
T = size(Y, 2);
C = zeros(T, 1);
alpha = zeros(K, T);
for k = 1:K
    alpha(k, 1) = pi(k)* normpdf(Y(1),E.mu(k), E.sigma2(k)); % init alpha z1
end

C(1) = sum(alpha(:,1));
alpha(:, 1) = alpha(:,1)/C(1); % normalise to get posterior

for t = 2:T
    %   k x 1       k x 1         k x k,  k x 1
    Emis_prob = zeros(K,1);
    for k = 1:K
       Emis_prob(k) = normpdf(Y(t), E.mu(k), E.sigma2(k));
    end
    alpha(:, t) = Emis_prob.*(A'*alpha(:,t-1)); % iterate
                %  k x 1
    C(t) = sum(alpha(:,t));
    % k x 1          k x 1   1 x 1
    alpha(:,t) = alpha(:,t)./C(t); % renormalise
       error = 0.01;
    assert(abs( sum(alpha(:,t))-1.000)< error);
end

end