N  = 10;         % number of sequences
T  = 100;        % length of the sequence
pi = [0.5; 0.5]; % inital probability pi_1 = 0.5 and pi_2 =0.5

%%two states hence A is a 2X2 matrix 
A  = [0.4 0.6 ; 0.4 0.6 ];         %p(y_t|y_{t-1})

%%alphabet of 6 letters (e.g., a die with 6 sides) E(i,j) is the
E = [1/6 1/6 1/6 1/6 1/6 1/6;      %p(x_t|y_{t}) 
    1/10 1/10 1/10 1/10 1/10 1/2];

[ Y, S ] = HmmGenerateData(N, T, pi, A, E); 
% 10 x 100

%%Y is the set of generated observations 
%%S is the set of ground truth sequence of latent vectors 
pi_e = pi;
A_e = A;
E_e = E;
% pi_e = reshape([0.5, 0.5],2,1);
% A_e = repmat(0.5,2,2);
% E_e = repmat(1/6, 2, 6);

for iter = 1:100

  [E1, E3, sums] =  EM_HMM_discrete_E(N, pi_e, A_e, E_e, Y);
%  sums
  [bj, pi_e, A_e] = EM_HMM_discrete_M(N, T, size(pi, 1), size(E, 2), E1, E3, Y);
  E_e(2, 1:5) = bj;
  E_e(2,6) = 1-sum(bj);
  
  pi_e = reshape(pi_e, 2, 1);
  
end

pi_e
A_e
E_e