N  = 10;         % number of sequences
T  = 100;        % length of the sequence
pi = [0.5; 0.5]; % inital probability pi_1 = 0.5 and pi_2 =0.5

%%two states hence A is a 2X2 matrix 
A  = [0.4 0.6 ; 0.4 0.6 ];         %p(y_t|y_{t-1})

%%alphabet of 6 letters (e.g., a die with 6 sides) E(i,j) is the
E = [1/6 1/6 1/6 1/6 1/6 1/6;      %p(x_t|y_{t}) 
    1/10 1/10 1/10 1/10 1/10 1/2];

[ Y, S ] = HmmGenerateData(N, T, pi, A, E); 
%%Y is the set of generated observations  size is 10 x 100
%%S is the set of ground truth sequence of latent vectors

%%%%%%%%%%%%%%% Use origin distribution data as initial data to run discrete HMM %%%%%%%%%%%%%%%
pi_e = pi;
A_e = A;
E_e = E;

%%%%%%%%%%%%%%% Use even distribution data as initial data to run discrete HMM %%%%%%%%%%%%%%%
% pi_e = reshape([0.5, 0.5],2,1);
% A_e = repmat(0.5,2,2);
% E_e = repmat(1/6, 2, 6);

%%%%%%%%%%%%%%%%%%%%%%%%%%% Start HMM process %%%%%%%%%%%%%%%%%%%%%%%%%%
K = size(A, 1);
NumObsers = size(E, 2);
IterationNum = 1000;

for iter = 1:IterationNum
  % Expectation step for discrete HMM
  [E1, E3, sums] =  EM_HMM_discrete_E(N, T, K, NumObsers, pi_e, A_e, E_e, Y);
  for k = 1:K
       er=0.01;
       % checking row sum of emission probability matrix is 1
       assert(abs(sum(E_e(k,:))-1.00)<er);
  end
  % Maximization step for discrete HMM
  [E_e, pi_e, A_e] = EM_HMM_discrete_M(N, T, K, NumObsers, E1, E3, Y);
  pi_e = reshape(pi_e, 2, 1);  
end

% Log result
pi_e
A_e
E_e

% Run viterbi decoding for dicrete HMM
S_e = EM_HMM_discrete_viterbi(N, T, K, pi_e, A_e, E_e, Y);
% Output accuracy rate of decoding
1 - (sum(sum(abs(S_e-S)))/(N*T))