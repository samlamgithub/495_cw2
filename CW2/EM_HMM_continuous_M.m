function [ mu, sigma, pi, a ] = EM_HMM_continuous_M(N, T, K, E1, E3, Y, E_e )
  %      kx1, kx1, k x 1, k x K        /      N x T x K, N x T-1 x K x K, N x T

  mu = zeros(K, 1);
  sigma = zeros(K, 1);
  for k = 1:K
       mu(k) = sum(sum(E1(:,:,k).*Y))/sum(sum(E1(:,:,k)));
       pre_mu = reshape(E_e.mu, K, 1);
       for n = 1:N
           for t = 1:T
               sigma(k) = sigma(k) + E1(n,t,k)*power((Y(n,t)-pre_mu(k)),2);
           end
       end
       sigma(k) = sigma(k)/sum(sum(E1(:,:,k)));
  end
  
  pi = reshape(sum(E1(:,1,:)),K,1);
  pi = pi/sum(sum(E1(:,1,:)));
  
  a = zeros(K, K);
  for j = 1:K
    a(j, :) = reshape(sum(sum(E3(:,:,j,:))), 1, K);
    a(j, :) = a(j, :) ./ sum(sum(sum(E3(:,:,j,:))));
  end
  
  
  
  
  
  
  
  

