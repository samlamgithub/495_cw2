function [ E1, E3, sums ] = EM_HMM_continuous_E(N, pi, A, E, Y)
  %   N x T x K, N x T-1 x K x K, N x T-1     k x 1, k x k, k x 2, N x T
  
T = size(Y, 2); % num seq
K = size(pi, 1); % num latent state

alphas = zeros(N, T, K);
betas = zeros(N, T, K);
C = zeros(N, T);

for n = 1:N
    [an, cn] = continuous_filtering(pi, A, E, Y(n, :));
	alphas(n, :, :) = an';
    C(n, :) = cn';
	betas(n, :, :) = continuous_smoothing(pi, A, E, Y(n, :), cn)';
end

% p_total = sum(C, 2); % nx1

% N x T x K  
E1 = alphas.*betas;

E3 = zeros(N, T-1, K, K);
sums = zeros(N, T-1);
for n = 1:N
    for t = 2:T
        for j = 1:K
            for k = 1:K
                p = normpdf(Y(n, t), E.mu(k), sqrt(E.sigma2(k)));
                E3(n, t-1, j, k) = alphas(n, t-1, j) * p * A(j, k) * betas(n,t,k) / C(n, t);              
            end
        end
        sums(n,t) = sum(sum(E3(n, t-1, :, :)));
        er=0.01;
       assert(abs(sums(n,t)-1.00)<er);
    end
%     E3(n,:,:,:) = E3(n,:,:,:)./p_total(n);
end