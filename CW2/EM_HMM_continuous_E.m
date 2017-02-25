function [ postLatent, postTransi, sums ] = EM_HMM_continuous_E(N, T, K, pi, A, E, Y)
% Out size:N x T x K, N x T-1 x K x K, N x T-1           In size: k x 1, k x k, k x 2, N x T

alphas = zeros(N, T, K); % init alphas
betas = zeros(N, T, K); % init betas
C = zeros(N, T); % init scaling factor

% Calculating all the alphas, betas and scaling factors
for n = 1:N
    [an, cn] = continuous_filtering(T, K, pi, A, E, Y(n, :));
	alphas(n, :, :) = an';
    C(n, :) = cn';
	betas(n, :, :) = continuous_smoothing(T, K, A, E, Y(n, :), cn)';
end

% size is N x T x K  
postLatent = alphas.*betas;

postTransi = zeros(N, T-1, K, K);  % init post transition prob
sums = zeros(N, T-1);
for n = 1:N
    for t = 2:T
        for j = 1:K
            for k = 1:K
                p = normpdf(Y(n, t), E.mu(k), sqrt(E.sigma2(k)));
                % calculating post transition prob
                postTransi(n, t-1, j, k) = alphas(n, t-1, j) * p * A(j, k) * betas(n,t,k) / C(n, t);              
            end
        end
        % assert sum of posterior transition prob is 1
        sums(n,t) = sum(sum(postTransi(n, t-1, :, :)));
        er=0.01;
        assert(abs(sums(n,t)-1.00)<er);
    end
end