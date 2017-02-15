function [ alpha_zt ] = Filtering(T, pi, A, E, Y, t)

[Num_state, Num_obser] = size(E);
Seq_Len = size(Y, 2);

alpha = zeros(Num_state, t);
for k = 1:Num_state
    alpha(k, 1) = pi(k) * E(k, Y(1)); % init alpha z1
end
alpha(:,1) = alpha(:,1)./sum(alpha(:,1)); % normalise to get posterior
    
for k = 2:t
    alpha(:, k) = E(:, Y(k)).*(A*alpha(:,k-1)); % iterate
    alpha(:,k) = alpha(:,k)./sum(alpha(:,k)); % renormalise
end

return alpha;

end