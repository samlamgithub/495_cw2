function [ alpha, C ] = discrete_filtering(pi, A, E, Y)
%                             k x 1, k x k, k x 6, 1 x T
K = size(pi, 1);
T = size(Y, 2);
C = zeros(T, 1);
alpha = zeros(K, T);
alpha(:, 1) = pi.* E(:, Y(1)); % init alpha z1
C(1) = sum(alpha(:,1));
alpha(:, 1) = alpha(:,1)/C(1); % normalise to get posterior

for t = 2:T
    %   k x 1       k x 1         k x k,  k x 1
    alpha(:, t) = E(:, Y(t)).*(A'*alpha(:,t-1)); % iterate
                %  k x 1
    C(t) = sum(alpha(:,t));
    % k x 1          k x 1   1 x 1
    alpha(:,t) = alpha(:,t)./C(t); % renormalise
%     sum(alpha(:,t))
    error = 0.01;
    assert(abs(sum(alpha(:,t))-1.000)< error);
    1;
end

end
