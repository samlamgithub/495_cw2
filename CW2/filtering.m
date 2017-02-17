function [ alpha, C ] = Filtering(pi, A, E, Y)

K = size(pi, 1);
T = size(Y, 2);
C = zeros(T);
alpha = zeros(K, T);
alpha(:, 1) = pi.* E(:, Y(1)); % init alpha z1
C(1) = sum(alpha(:,1));
alpha(:, 1) = alpha(:,1)/C(1)); % normalise to get posterior

for k = 2:T
    alpha(:, k) = E(:, Y(k)).*(A*alpha(:,k-1)); % iterate
    C(k) = sum(alpha(:,k));
    alpha(:,k) = alpha(:,k)./C(k); % renormalise
end

end