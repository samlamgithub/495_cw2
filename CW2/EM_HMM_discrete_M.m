function [ b2, pi, a ] = EM_HMM_discrete_M(N, T, K, NumObsers, E1, E3, Y )
  %        5 x 1, k x 1, k x K        /    N x T xK,  N x T x K, N x T-1 x K x K, N x T
  
  b2 = zeros(NumObsers - 1, 1);
  for n = 1:N
    for t = 1:T
        for j = 1:NumObsers - 1
             b2(j) = b2(j) + E1(n, t, 2)* (Y(n, t) == j);
        end
    end
  end
  b2 = b2./(sum(sum(E1(:,:,2))));
  
  pi = reshape(sum(E1(:,1,:)),K,1);
  pi = pi./sum(sum(E1(:,1,:)));
  
  a = zeros(K, K);
  for j = 1:K
    a(j, :) = reshape(sum(sum(E3(:,:,j,:))), 1, K);
    a(j, :) = a(j, :) ./ sum(sum(sum(E3(:,:,j,:))));
  end
  
  
  
  
  
  
  
  

