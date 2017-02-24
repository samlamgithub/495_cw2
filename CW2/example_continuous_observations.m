clc;
clear all;

N  = 100;         % number of sequences
T  = 100;        % length of the sequence
pi = [0.5; 0.5]; % inital probability pi_1 = 0.5 and pi_2 =0.5

%%two states hence A is a 2X2 matrix 
A  = [0.4 0.6 ; 0.4 0.6 ];         %p(y_t|y_{t-1})

pi
%%one dimensional Gaussians 

E.mu    =[ .1  .5]; %%the means of each of the Gaussians
E.sigma2=[ .4 .8]; %%the variances

[ Y, S ] = HmmGenerateData(N, T, pi, A, E, 'normal'); 
%%Y is the set of generated observations, size is 10 x 100
%%S is the set of ground truth sequence of latent vectors

%%%%%%%%%%%%%%% Use origin distribution data as initial data to run continuous HMM %%%%%%%%%%%%%%%
pi_e = pi;
A_e = A;
E_e = E;

%%%%%%%%%%%%%%% Use random distribution data as initial data to run continuous HMM %%%%%%%%%%%%%%%
% pi_e = reshape([0.4, 0.6],2,1);
% A_e = [0.2, 0.8;0.2,0.8];
% E_e.mu    =[ .2 , .9]; %%the means of each of the Gaussians
% E_e.sigma2=[ .1 , .8]; %%the variances

%%%%%%%%%%%%%%%%%%%%%%%%%%% Start HMM process %%%%%%%%%%%%%%%%%%%%%%%%%%
K = size(A, 1);
IterationNum = 1000;

for iter = 1:IterationNum
  % Expectation step for discrete HMM
  [E1, E3, sums] =  EM_HMM_continuous_E(N, T, K, pi_e, A_e, E_e, Y);
  % Maximization step for discrete HMM
  [mu, sigma2, pi_e, A_e] = EM_HMM_continuous_M(N, T, K, E1, E3, Y, E_e);
   reshape(mu, 1, 2);
   reshape(sigma2, 1, 2);
   E_e.mu = mu;
   E_e.sigma2 = sigma2;
   pi_e = reshape(pi_e, 2, 1);
end

pi_e
A_e
E_e.mu
E_e.sigma2

% Run viterbi decoding for dicrete HMM
S_e = EM_HMM_continuous_viterbi(N, T, K, pi_e, A_e, E_e, Y);
% Output accuracy rate of decoding
1 - (sum(sum(abs(S_e-S)))/(N*T))

