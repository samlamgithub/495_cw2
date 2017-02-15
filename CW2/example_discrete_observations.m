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
pi_e = [0.5, 0.5];
A_e = repmat(0.5,2,2);
E_e = repmat(1/12, 2, 6);

for iter = 1:20

[E_z1, E_ztl, E3] =  EM_HMM_discrete_E(N, T, pi_e, A_e, E_e, Y);
 
  [bj2, pi_e, A_e ] = EM_HMM_discrete_M(N, T, size(pi), size(E, 2), E_z1, E_ztl, E3, Y);
  p = sum(bj2)/(size(E,2)-1);
  pi_e = [1-p, p];
  
end