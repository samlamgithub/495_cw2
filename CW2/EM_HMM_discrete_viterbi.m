function [ S_e ] = EM_HMM_discrete_viterbi(N, T, K, pi_e, A_e, E_e , Y)
% Out size: N x T                     In size:    K x 1, K x K, K x NumObsers, N x T

S_e = zeros(N, T); % init latent variables

% take log of these matrixes for calculation
logPi = log2(pi_e);
logA = log2(A_e);
logE = log2(E_e);

for n = 1:N
    delta_probs = zeros(T, K);
    delta_probs(1, :) = logPi' + logE(:, Y(n, 1))'; % init the first observation prob
    [nothing, S_e(n,1)] = max(delta_probs(1, :)); % init first latent
    for t = 2:T
         for k = 1:K
               % pro of end up in latent j at time t: from latent i at time t-1
               % delta(t, j) = C_t^-1 x p(x_t|ztj=1) max for i: delta(t-1,
               % i) x A(i to j)
             delta_probs(t, k) = logE(k, Y(n, t)) + max(delta_probs(t-1, :)' + logA(:, k));
         end
         % keep track of max porob latent
         [nothing, S_e(n, t)] = max(delta_probs(t, :));
    end 
end
end


