functions{
real h_lambda(real lambda1, real lambda2, real eta){
 if (eta > 0){
	if (lambda1 > 0)
		return (exp(lambda1 * eta) - 1) / lambda1;
	else if (lambda1 == 0)
		return eta;
	else //alpha1 < 0
		return -log(1 - lambda1 * eta) / lambda1;
  }
  else{ //eta <= 0
	 if (lambda2 > 0)
		return -(exp(-lambda2 * eta) - 1) / lambda2;
	else if (lambda2 == 0)
		return eta;
	else //alpha2 < 0
		return log(1 + lambda2 * eta) / lambda2;
  }
 }
}

data{
int<lower=0> n_student;
int<lower=0> n_item;
int<lower=0,upper=1> Y[n_student,n_item];

}

parameters{
 vector[n_student] theta;
 vector[n_item] beta;
 vector<lower=-1>[n_item] lambda1;
 vector<lower=-1>[n_item] lambda2;
 real<lower=0> sig_beta;
 real<lower=0> sig_lambda;
}

model{
 theta ~ normal(0,1);
 beta ~ normal(0,sig_beta);
 lambda1 ~ normal(0,sig_lambda); 
 lambda2 ~ normal(0,sig_lambda); 
 sig_beta ~ cauchy(0,5);
 sig_lambda ~ cauchy(0,5);
 
 for(i in 1:n_student){
  for (j in 1:n_item){
    Y[i,j] ~ bernoulli_logit( h_lambda(lambda1[j], lambda2[j], (theta[i] - beta[j])));   
  }
 }
}

generated quantities{
  vector[n_item] log_lik_Y[n_student];
  
 for (i in 1: n_student){
	for (j in 1: n_item){
	  log_lik_Y[i,j] = bernoulli_logit_lpmf(Y[i,j] | h_lambda(lambda1[j], lambda2[j], (theta[i] - beta[j])));
	  }
  }
}

