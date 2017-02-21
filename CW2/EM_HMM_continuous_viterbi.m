function [ S_e ] = EM_HMM_continuous_viterbi( N, T, K, pi_e, A_e, E_e , Y)
%         N x T                         K x 1, K x K, K x 2, N x T

S_e = zeros(N, T);
% log2(x)
%  [a, b] = max([3,4,5])

logPi = log2(pi_e);
logA = log2(A_e);

for n = 1:N
    
    delta_probs = zeros(T, K);
    for k = 1:K
         delta_probs(1,k) = logPi(k) + log2(normpdf(Y(n, 1), E_e.mu(k), sqrt(E_e.sigma2(k))));
    end
    [nothing, S_e(n,1)] = max(delta_probs(1, :));
    
    for t = 2:T
         for k = 1:K
               % pro of end up in latent j at time t: from latent i at time t-1
               % delta(t, j) = C_t^-1 x p(x_t|ztj=1) max for i: delta(t-1,
               % i) x A(i to j)
             delta_probs(t, k) = log2(normpdf(Y(n, t), E_e.mu(k), sqrt(E_e.sigma2(k)))) + max(delta_probs(t-1, :)' + logA(:, k));
%              k2
%              logE(k, Y(n, t))  + delta_probs(t-1, k2)' + logA(k2, k)
         end
         [nothing, S_e(n, t)] = max(delta_probs(t, :));
    end 
end
end


