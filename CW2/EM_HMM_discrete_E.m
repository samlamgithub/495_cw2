function [ E1, E3, sums ] = EM_HMM_discrete_E(N, T, K, NumObsers, pi, A, E, Y)
%output size: N x T x K, N x T-1 x K x K, N x T-1

alphas = zeros(N, T, K);
betas = zeros(N, T, K);
C = zeros(N, T);

for n = 1:N
    [an, cn] = discrete_filtering(pi, A, E, Y(n, :));
	alphas(n, :, :) = an';
    C(n, :) = cn';
	betas(n, :, :) = discrete_smoothing(pi, A, E, Y(n, :), C(n, :)')';
end

E1 = alphas.*betas;
%  size E1 is (N,T,K)

E3 = zeros(N, T-1, K, K);
sums = zeros(N, T-1);
for n = 1:N
    for t = 2:T
        for j = 1:K
            for k = 1:K
                p = 1;
                for r = 1:NumObsers
                    if Y(n, t) == r
                         p = p * E(k, r);
                    end
                end
                E3(n, t-1, j, k) = alphas(n, t-1, j) * p * A(j, k) * betas(n,t,k) / C(n, t);              
            end
        end
        sums(n,t) = sum(sum(E3(n, t-1, :, :)));
        er=0.01;
        assert(abs(sums(n,t)-1.00)<er);
    end
end