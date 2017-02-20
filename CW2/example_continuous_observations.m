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
%%Y is the set of generated observations 
%%S is the set of ground truth sequence of latent vectors 

%10 x 100

pi_e = reshape([0.4, 0.6],2,1);
A_e = [0.7 0.3 ; 0.7 0.3 ];    
E_e.mu    =[ .2 , 3]; %%the means of each of the Gaussians
E_e.sigma2=[ .5 , .6]; %%the variances

for iter = 1:1000
  [E1, E3, sums] =  EM_HMM_continuous_E(N, pi_e, A_e, E_e, Y);
%  sums
  [E_e.mu, E_e.sigma2, pi_e, A_e] = EM_HMM_continuous_M(N, T, size(pi, 1), E1, E3, Y, E_e);
   reshape(E_e.mu, 1, 2);
   reshape(E_e.sigma2, 1, 2);
  pi_e = reshape(pi_e, 2, 1);  
%   A_e
% pi_e
% A_e
% E_e.mu
% E_e.sigma2
end

pi_e
A_e
E_e.mu
E_e.sigma2

