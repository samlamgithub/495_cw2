function [ E1, E3 ] = EM_HMM_discrete_E(N, pi, A, E, Y)
  %        N x K,  N x T x K, N x T-1 x K x K
  
T = size(Y, 2); % num seq
K = size(pi, 1); % num latent state
NumObsers = size(E, 2);

alphas = zeros(N, T, K);
betas = zeros(N, T, K);
C = zeros(N, T);

for n = 1:N
    [an, cn] = filtering(pi, A, E, Y(n, :));
	alphas(n, :, :) = an';
    C(n, :) = cn';
	betas(n, :, :) = smoothing(pi, A, E, Y(n, :), C(n, :)')';
end

p_total = sum(C, 2); % nx1

E1 = zeros(N,T,K);

for n = 1:N
    for t = 1:T
        E1(n,t, :) = alphas(n,t,:).*betas(n,t,:);
    end
    E1(n,:) = E1(n,:)./p_total(n);
end

E3 = zeros(N, T-1, K, K);

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
                E3(n, t-1, j, k) = alphas(n, t-1, j) * p * A(j, k) * betas(n,t,k);
            end
        end
    end
    E3(n,:,:,:) = E3(n,:,:,:)./p_total(n);
end