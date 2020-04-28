#include <cmath>
#include <math.h>
#include <random>
#include <RcppArmadillo.h>
using namespace Rcpp;
using namespace std;

// [[Rcpp::depends(RcppArmadillo)]]

double loglambdaTarget(double lambda, vector<double> tau2, double a, double b) {
  return((-a-1)*log(lambda)-(pow(lambda,2)/2*accumulate(tau2.begin(), tau2.end(), 0)-1/(b*lambda)));
}

double logtau2jTarget(double tau2j, double lambda, double betaj){
  return(-log(sqrt(tau2j)) - 1.0/2.0*(1.0/tau2j*pow(betaj,2) + pow(lambda,2)*tau2j));
}

double lambdaDraw(double current, vector<double> tau2, double a, double b){
  //std::random_device rd;
  //std::mt19937 mt(rd());
  //std::uniform_real_distribution<double> dist(0, 1.0);
  // std::normal_distribution<double> norm(0, current);
  double e = arma::as_scalar(arma::randn(1) * sqrt(current));
  double u = arma::as_scalar(arma::randu(1));
  double proposed = exp(log(current)+e);
  double logr = loglambdaTarget(proposed,tau2,1.0,b)-
    loglambdaTarget(current,tau2,1.0,b)+log(proposed)-
    log(current);
  if(log(u)<logr){
    current = proposed;
  }
  return current;
}

vector<double> tau2Draw(vector<double> current, vector<double> beta, double lambda){
  // std::random_device rd;
  // std::mt19937 mt(rd());
  // std::uniform_real_distribution<double> dist(0, 1.0);
  // std::normal_distribution<double> norm(0, 1);
  double u = arma::as_scalar(arma::randu(1));
  double logr = 0;
  double tau2j_proposed = 0;

  for(int j=0;j < current.size(); j++){
    tau2j_proposed = exp(log(current[j]+arma::as_scalar(arma::randn(1))));
    logr =
      logtau2jTarget(tau2j_proposed, lambda, beta[j]) -
      logtau2jTarget(current[j], lambda, beta[j]) +
      log(tau2j_proposed)-log(current[j]);
    if(log(u)<logr){
      current[j] = tau2j_proposed;
    }
  }
  return current;
}

double sigma2Draw(const arma::vec & y, const arma::mat & X){
  // std::random_device rd;
  // std::mt19937 mt(rd());
  int n = X.n_rows;
  arma::colvec coef = arma::solve(X, y);
  arma::colvec resid = y - X*coef;
  double rss = arma::as_scalar(arma::trans(resid)*resid);
  arma::mat v = arma::randg(1,1, arma::distr_param(n/2.0+0.1,1.0/(rss/2.0+0.1)));
  //std::gamma_distribution<double> gamma(n/2.0+0.1,1.0/(rss/2.0+0.1));
  return 1.0/arma::as_scalar(v);
}

List betaMeanCov(const arma::vec & y, const arma::mat & X, arma::vec sigma2, arma::vec tau2){
  // std::random_device rd;
  // std::mt19937 mt(rd());
  int p = X.n_cols;
  arma::mat I = arma::eye(p,p);
  arma::mat tau2_inv = arma::inv(arma::diagmat(tau2));
  arma::mat M = arma::inv(X.t()*X/sigma2+tau2_inv);
  arma::colvec m = M*X.t()*y/sigma2;
  return List::create(Named("cov") = M, Named("mean")= m);
}

vector<double> rmvnorm_cpp(int n, const arma::vec& mu, const arma::mat& Sigma) {
  unsigned int p = Sigma.n_cols;
  // First draw N x P values from a N(0,1)
  Rcpp::NumericVector draw = Rcpp::rnorm(n*p);
  // Instantiate an Armadillo matrix with the // drawn values using advanced constructor // to reuse allocated memory
  arma::mat Z = arma::mat(draw.begin(), n, p,
                          false, true); // Simpler, less performant alternative
  // arma::mat Z = Rcpp::as<arma::mat>(draw);
  // Generate a sample from the Transformed // Multivariate Normal
  arma::mat Y = arma::repmat(mu, 1, n).t() + Z * arma::chol(Sigma);
  return arma::conv_to<vector<double>>::from(Y);
}

// [[Rcpp::export]]
List mcmc(double numSamples, const arma::vec & y, const arma::mat & X){
  int p = X.n_cols;
  // 
  arma::vec lambdaChain = arma::ones(numSamples);
  arma::vec sigma2Chain = arma::ones(numSamples);
  arma::mat tau2Chain(p,numSamples);
  arma::mat betaChain(p,numSamples);

  // std::random_device rd;
  // std::mt19937 mt(rd());
  
  arma::mat I = arma::eye(p,p);
  arma::mat tau2_inv = arma::eye(p,p);
  arma::mat M = arma::eye(p,p);
  arma::colvec m = arma::ones(p);

  
  double lambda = 1;
  double sigma2 = 1;
  vector<double> beta(p, 0.0);
  vector<double> tau2(p, 1.0);
  
  for(int s=0; s<numSamples; s++){
    lambda = lambdaDraw(lambda, tau2, 1, 1);
    tau2 = tau2Draw(tau2, beta, lambda);
    sigma2 = sigma2Draw(y, X);
    arma::vec tau2_(tau2);
    tau2_inv = arma::inv(arma::diagmat(tau2_));
    M = arma::inv(X.t()*X/sigma2+tau2_inv);
    m = M*X.t()*y/sigma2;
    beta = rmvnorm_cpp(1,m,M);
    
    lambdaChain[s] = lambda;
    sigma2Chain[s]= sigma2;
    arma::colvec armaTau2(tau2);
    tau2Chain.insert_cols(s, armaTau2);
    arma::colvec armaBeta(beta);
    betaChain.insert_cols(s, armaBeta);
  }
  return List::create(Named("beta.chain") = betaChain, Named("sigma2.chain")= sigma2Chain,
                      Named("tau2.chain") = tau2Chain, Named("lambda.chain") = lambdaChain);
}
