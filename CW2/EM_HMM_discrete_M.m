function [ E_e, pi, a ] = EM_HMM_discrete_M(N, T, K, NumObsers, E1, E3, Y )
  %    K x NumObsers   , k x 1, k x K        /    N x T x K, N x T-1 x K x K, N x T
  
  E_e = zeros(K, NumObsers); % New emission probabilities
  for k = 1:K
      for j = 1:NumObsers
          for n = 1:N
            for t = 1:T
               E_e(k, j) = E_e(k, j) + E1(n, t, k) * (Y(n, t) == j);
            end
          end
      end
      E_e(k, :) = E_e(k, :) / sum(E_e(k, :));
  end
  
  pi = reshape(sum(E1(:,1,:)),K,1);
  pi = pi./sum(sum(E1(:,1,:)));
  
  a = zeros(K, K);
  for j = 1:K
    a(j, :) = reshape(sum(sum(E3(:,:,j,:))), 1, K);
    a(j, :) = a(j, :) ./ sum(sum(sum(E3(:,:,j,:))));
  end
  
  
  
  
  
  
  
  

