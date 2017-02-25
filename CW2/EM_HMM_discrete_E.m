function [ postLatent, posttransi, sums ] = EM_HMM_discrete_E(N, T, K, NumObsers, pi, A, E, Y)
% Out size: N x T x K, N x T-1 x K x K, N x T-1

alphas = zeros(N, T, K); % init alphas
betas = zeros(N, T, K); % init betas
C = zeros(N, T); % init scaling factor

% Calculating all the alphas, betas and scaling factors
for n = 1:N
    [an, cn] = discrete_filtering(T, K, pi, A, E, Y(n, :));
	alphas(n, :, :) = an';
    C(n, :) = cn';
	betas(n, :, :) = discrete_smoothing(T, K, A, E, Y(n, :), C(n, :)')';
end

postLatent = alphas.*betas;
%  size E1 is (N,T,K)

posttransi = zeros(N, T-1, K, K); % init post transition prob
sums = zeros(N, T-1);
for n = 1:N
    for t = 2:T
        for j = 1:K
            for k = 1:K
                p = 1;
                for r = 1:NumObsers
                   p = p * power(E(k, r), Y(n, t) == r);
                end
                % calculating post transition prob
                posttransi(n, t-1, j, k) = alphas(n, t-1, j) * p * A(j, k) * betas(n,t,k) / C(n, t);              
            end
        end
        sums(n,t) = sum(sum(posttransi(n, t-1, :, :)));
        er=0.01;
        % assert sum of posterior transition prob is 1
        assert(abs(sums(n,t)-1.00)<er);
    end
end