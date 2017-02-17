function [ E1, E3 ] = EM_HMM_discrete_E(N, pi, A, E, Y)
  %        N x K,  N x T x K, N x T-1 x K x K
  
T = size(Y, 2); % num seq
K = size(pi, 1); % num latent state
NumObsers = size(E, 2);

alphas = zeros(N, T, K);
betas = zeros(N, T, K);
C = zeros(N, T);

for n = 1:N
	alphas(n, :, :), C(n, :) = Filtering(pi, A, E, Y(n, :))';
	betas(n, :, :) = Smoothing(pi, A, E, Y(n, :), C(n, :)')';
end

p_total = sum(C, 2); % nx1

E1 = zeros(N,T,K);

for n = 1:N
    for t = 1:T
        E1(n,t, :) = alphas(n,t,:).*betas(n,t,:);
    end
    E1(n,:) = E1(n,:)./p_total(n);
end

E3 = zeros(N, T-1, K, K);

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
                E3(n, t-1, j, k) = alphas(n, t-1, j) * p * A(j, k) * betas(n,t,k);
            end
        end
    end
    E3(n,:,:,:) = E3(n,:,:,:)./p_total(n);
end



function [ b2, pi, a ] = EM_HMM_discrete_M(N, T, K, NumObsers, E1, E3, Y )
  %        5 x 1, k x 1, k x K        /    N x K,  N x T x K, N x T-1 x K x K
  
  b2 = zeros(NumObsers - 1, 1);
  for n = 1:N
    for t = 1:T
        for j = 1:NumObsers - 1
            if Y(n, t) == j
                b2(j) = b2(j) + E1(n, t, 2);
            end 
        end
    end
  end
  b2 = b2./(sum(sum(E1(:,:,2))));
  
  pi = sum(E1(:,1,:))';
  pi = pi./sum(sum(E1(:,1,:)));
  
  a = zeros(K, K);
  for j = 1:K
    a(j, :) = sum(sum(E3(:,:,j,:)))';
    a(j, :) = a(j, :) ./ sum(sum(sum(E3(:,:,j,:))));
  end
  
  
  
  
  
  
  
  
