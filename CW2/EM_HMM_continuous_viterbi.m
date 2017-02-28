function [ S_e ] = EM_HMM_continuous_viterbi(N, T, K, pi_e, A_e, E_e , Y)
% Out size: N x T                          In size: K x 1, K x K, K x 2, N x T

S_e = zeros(N, T); % init latent variables

% take log of these matrixes for calculation
logPi = log2(pi_e);
logA = log2(A_e);

for n = 1:N
    delta_probs = zeros(T, K);
    for k = 1:K
        % init the first observation prob
        delta_probs(1,k) = logPi(k) + log2(normpdf(Y(n, 1), E_e.mu(k), sqrt(E_e.sigma2(k))));
    end
    % init first latent
    Z(1, :) = [1, 2];
    
    for t = 2:T
         for k = 1:K
               % pro of end up in latent j at time t: from latent i at time t-1
               % delta(t, j) = C_t^-1 x p(x_t|ztj=1) max for i: delta(t-1,
               % i) x A(i to j)
             [maxP, maxZ] = max(delta_probs(t-1, :)' + logA(:, k))
             Z(t, k) = maxZ;
             delta_probs(t, k) = log2(normpdf(Y(n, t), E_e.mu(k), sqrt(E_e.sigma2(k)))) + maxP;
         end
    end
    % backtrack
    [nothing, S_e(n, T)] = max(delta_probs(T, :));
    for t=T-1:-1:1
          S_e(n, t) = Z(t+1, S_e(n, t+1));
    end
end
end
