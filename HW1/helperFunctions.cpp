#include <cmath>
#include <math.h>
#include <random>
#include <RcppArmadillo.h>
using namespace Rcpp;
using namespace std;

// [[Rcpp::depends(RcppArmadillo)]]

const arma::mat loglambdaTarget(const arma::mat lambda, const arma::vec tau2, double a, double b) {
  return((-a-1)*log(lambda)-(pow(lambda,2)/2*arma::accu(tau2)-1/(b*lambda)));
}

const arma::mat logtau2jTarget(const arma::mat tau2j, const arma::mat lambda, const arma::mat betaj){
  return(-log(sqrt(tau2j)) - 1.0/2.0*(1.0/tau2j*pow(betaj,2) + pow(lambda,2)*tau2j));
}

const arma::mat lambdaDraw(const arma::mat current, const arma::vec tau2, double a, double b){
  //std::random_device rd;
  //std::mt19937 mt(rd());
  //std::uniform_real_distribution<double> dist(0, 1.0);
  // std::normal_distribution<double> norm(0, current);
  const arma::mat e = arma::randn(1) * sqrt(current);
  const arma::mat u = arma::randu(1);
  const arma::mat proposed = exp(log(current)+e);
  const arma::mat logr = loglambdaTarget(proposed,tau2,1,b)-
    loglambdaTarget(current,tau2,1,b)+log(proposed)-
    log(current);
  bool go=log(u)<logr;
  if(go){
    current = proposed;
  }
  return current;
}

const arma::vec tau2Draw(const arma::vec current, const arma::vec beta, const arma::mat lambda){
  // std::random_device rd;
  // std::mt19937 mt(rd());
  // std::uniform_real_distribution<double> dist(0, 1.0);
  // std::normal_distribution<double> norm(0, 1);
  const arma::mat e = arma::randn(1);
  const arma::mat u = arma::randu(1);
  const arma::mat proposed = exp(log(current)+e);
  
  for(int j=0;j < current.size(); j++){
    double tau2j_proposed = exp(log(current[j])+norm(mt));
    double logr =
      logtau2jTarget(tau2j_proposed, lambda, beta[j]) -
      logtau2jTarget(current[j], lambda, beta[j]) +
      log(tau2j_proposed)-log(current[j]);
    if(log(dist(u)) < logr){
      current[j] = tau2j_proposed;
    }
  }
  return current;
}

const arma::mat sigma2Draw(const arma::vec & beta, const arma::vec & y, const arma::mat & X){
  std::random_device rd;
  std::mt19937 mt(rd());
  int n = X.n_rows;
  arma::colvec coef = arma::solve(X, y);
  arma::colvec resid = y - X*coef;
  double rss = arma::as_scalar(arma::trans(resid)*resid);
  std::gamma_distribution<double> gamma(n/2.0+0.1,1.0/(rss/2.0+0.1));
  return 1.0/gamma(mt);
}

List betaMeanCov(const arma::vec & y, const arma::mat & X, double sigma2, const arma::vec & tau2){
  std::random_device rd;
  std::mt19937 mt(rd());
  int p = X.n_cols;
  arma::mat I = arma::eye(p,p);
  arma::mat tau2_inv = arma::inv(arma::diagmat(tau2));
  arma::mat M = arma::inv(X.t()*X/sigma2+tau2_inv);
  arma::colvec m = M*X.t()*y/sigma2;
  return List::create(Named("cov") = M, Named("mean")= m);
}

arma::mat rmvnorm_cpp(int n,
                  const arma::vec& mu,
                  const arma::mat& Sigma) {
  unsigned int p = Sigma.n_cols;
  // First draw N x P values from a N(0,1)
  Rcpp::NumericVector draw = Rcpp::rnorm(n*p);
  // Instantiate an Armadillo matrix with the // drawn values using advanced constructor // to reuse allocated memory
  arma::mat Z = arma::mat(draw.begin(), n, p,
                          false, true); // Simpler, less performant alternative
  // arma::mat Z = Rcpp::as<arma::mat>(draw);
  // Generate a sample from the Transformed // Multivariate Normal
  arma::mat Y = arma::repmat(mu, 1, n).t() +
    Z * arma::chol(Sigma);
  return Y;
}

// [[Rcpp::export]]
List mcmc(double numSamples, const arma::vec & y, const arma::mat & X){
  int p = X.n_cols;
  
  arma::vec lambdaChain = arma::ones(numSamples);
  arma::vec sigma2Chain = arma::ones(numSamples);
  arma::mat tau2Chain = arma::zeros(p,numSamples);
  arma::mat betaChain = arma::zeros(p,numSamples);
  
  std::random_device rd;
  std::mt19937 mt(rd());
  arma::mat I = arma::eye(p,p);
  arma::mat tau2_inv = arma::eye(p,p);
  arma::mat M = arma::eye(p,p);
  arma::colvec m = arma::ones(p);
  
  double lambda = 1; double sigma2 = 1;
  const arma::vec beta=arma::zeros(p); const arma::vec tau2=arma::ones(p);
  
  for(int s=0; s<numSamples; s++){
    lambda = lambdaDraw(lambda, tau2, 1, 1);
    tau2 = tau2Draw(tau2, beta, lambda);
    sigma2 = sigma2Draw(beta, y, X);
    tau2_inv = arma::inv(arma::diagmat(tau2));
    M = arma::inv(X.t()*X/sigma2+tau2_inv);
    m = M*X.t()*y/sigma2;
    beta = rmvnorm_cpp(1,m,M);
  }
  return List::create(Named("beta.chain") = betaChain, Named("sigma2.chain")= sigma2Chain,
                      Named("tau2.chain") = tau2Chain, Named("lambda.chain") = lambdaChain);
}