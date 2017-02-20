function [ E1, E3, sums ] = EM_HMM_discrete_E(N, pi, A, E, Y)
  %     N x T x K, N x T-1 x K x K, N x T-1
  
T = size(Y, 2); % num seq
K = size(pi, 1); % num latent state
NumObsers = size(E, 2);

alphas = zeros(N, T, K);
betas = zeros(N, T, K);
C = zeros(N, T);

for n = 1:N
    [an, cn] = discrete_filtering(pi, A, E, Y(n, :));
	alphas(n, :, :) = an';
    C(n, :) = cn';
	betas(n, :, :) = discrete_smoothing(pi, A, E, Y(n, :), C(n, :)')';
end

% p_total = sum(C, 2); % nx1

E1 = zeros(N,T,K);

E1(:, :, :) = alphas(:, :, :).*betas(:, :, :);
%     E1(n,:) = E1(n,:)./p_total(n);

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
                E3(n, t-1, j, k) = alphas(n, t-1, j) * p * A(j, k) * betas(n,t,k) * C(n, t);              
            end
        end
        sums(n,t) = sum(sum(E3(n, t-1, :, :)));
    end
%     E3(n,:,:,:) = E3(n,:,:,:)./p_total(n);
end