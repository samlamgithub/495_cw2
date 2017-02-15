function [ E_z1, E_ztl, E3 ] = EM_HMM_discrete_E(N, T, pi, A, E, Y)
  %        N x K,  N x T x K, N x T-1 x K x K
[Num_state, Num_obser] = size(E);
k = Num_state;

E_zl = zeros(N, K);

for n = 1:N
  p_total = sum(Filtering(T, pi, A, E, Y(n, :), T));
  E_zl(n,:) = Filtering(T, pi, A, E, Y(n, :), 1).*Smoothing(T, pi, A, E, Y(n, :), 1);
  E_zl(n,:) = E_zl(n,:)./p_total;
end

E_ztl = zeros(N, T, K);

for n = 1:N
    p_total = sum(Filtering(T, pi, A, E, Y(n, :), T));
    for t = 1:T
        E_ztl(n, t, :) = Filtering(T, pi, A, E, Y(n, :), t).*Smoothing(T, pi, A, E, Y(n, :), t);
    end
    E_ztl(n, :, :) = E_ztl(n, :, :)./p_total;
end

E3 = zeros(N, T-1, K, K);

for n = 1:N
    p_total = sum(Filtering(T, pi, A, E, Y(n, :), T));
    for t = 2:T
        for j = 1:K
            for k = 1:K
                p = 0;
                for r = 1:Num_state
                    if Y(n, t) == r
                         p = p * E(k, r);
                    end
                end
                E3(n, t-1, j, k) = Filtering(T, pi, A, E, Y(n, :), t-1)(j) * p * A(j, k) .* Smoothing(T, pi, A, E, Y(n, :), t)(k);
            end
        end
    end
    E3(n,:,:,:) = E3(n,:,:,:)./p_total;
end



function [ bj2, pi, a ] = EM_HMM_discrete_M(N, T, K, Num_state, E_z1, E_ztl, E3, Y )
  %        5 x 1, k x 1, k x K        /    N x K,  N x T x K, N x T-1 x K x K
  
  bj2 = zeros(Num_State - 1, 1);
  for n = 1:N
    for t = 1:T
        for j = 1:Num_State - 1
            if Y(n, t) == j
                bj2(j) = bj2(j) + E_ztl(n, t, 2);
            end 
        end
    end
  end
  bj2 = bj2./((sum(sum(E_ztl)))(2));
  
  pi = sum(E_z1)';
  pi = pi./(sum(sum(E_z1)));
  
  a = zeros(K, K);
  for j = 1:K
    a(j, :) = sum(sum(E3))(j);
    a(j, :) = a(j, :) ./ (sum(sum(sum(E3)), 2)(j));
  end
  
  
  
  
  
  
  
  
