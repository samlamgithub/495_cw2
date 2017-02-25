function [ mu, sigma, pi, a ] = EM_HMM_continuous_M(N, T, K, postLatent, posttransi, Y, E_e )
% Out size: Kx1, kx1, k x 1, k x K          In size: N x T x K, N x T-1 x K x K, N x T

  % Init emission probabilities distribution
  mu = zeros(K, 1);
  sigma = zeros(K, 1);
  for k = 1:K
       mu(k) = sum(sum(postLatent(:,:,k).*Y))/sum(sum(postLatent(:,:,k)));
       pre_mu = reshape(E_e.mu, K, 1);
       for n = 1:N
           for t = 1:T
               % calculating emission probabilities distribution mu
               sigma(k) = sigma(k) + postLatent(n,t,k)*power((Y(n,t)-pre_mu(k)),2);
           end
       end
       % calculating emission probabilities distribution sigma
       sigma(k) = sigma(k)/sum(sum(postLatent(:,:,k)));
  end
  
  % calculating next pi estimation
  pi = reshape(sum(postLatent(:,1,:)),K,1);
  pi = pi/sum(sum(postLatent(:,1,:)));
  
  % calculating next transition prob estimation
  a = zeros(K, K);
  for j = 1:K
    a(j, :) = reshape(sum(sum(posttransi(:,:,j,:))), 1, K);
    a(j, :) = a(j, :) ./ sum(sum(sum(posttransi(:,:,j,:))));
  end
  
  
  
  
  
  

