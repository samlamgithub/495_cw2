function [ E_e, pi, a ] = EM_HMM_discrete_M(N, T, K, NumObsers, postLatent, posttransi, Y )
% Out size: K x NumObsers, k x 1, k x K         In size: N x T x K, N x T-1 x K x K, N x T
  
  E_e = zeros(K, NumObsers); % Init emission probabilities
  for k = 1:K
      for j = 1:NumObsers
          for n = 1:N
            for t = 1:T
               E_e(k, j) = E_e(k, j) + postLatent(n, t, k) * (Y(n, t) == j);
               % calculating emission prob
            end
          end
      end
      E_e(k, :) = E_e(k, :) / sum(E_e(k, :)); % normalise Emission prob
  end
  
  % calculating next pi estimation
  pi = reshape(sum(postLatent(:,1,:)),K,1);
  pi = pi./sum(sum(postLatent(:,1,:)));
  
  % calculating next transition prob estimation
  a = zeros(K, K);
  for j = 1:K
    a(j, :) = reshape(sum(sum(posttransi(:,:,j,:))), 1, K);
    a(j, :) = a(j, :) ./ sum(sum(sum(posttransi(:,:,j,:))));
  end
  
  
  
  
  
  
  
  

