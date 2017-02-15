function [ beta_zt ] = Smoothing(T, pi, A, E, Y, t)

[Num_state, Num_obser] = size(E);
T = Seq_Len = size(Y, 2);

beta = zeros(Num_state, T-t);
beta(:, T-t) = ones(Num_state, 1); % init beta zT

for k = T-1:t
    beta(:, k) = beta(:,k+1)'*A*E(:, Y(k+1)); % iterate
    beta(:,k) = beta(:,k)./sum(beta(:,k)); % renormalise
end

return beta;

end