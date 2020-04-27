#include <cmath>
#include <math.h>
#include <random>
#include <RcppArmadillo.h>
using namespace Rcpp;
using namespace std;

// [[Rcpp::depends(RcppArmadillo)]]

double loglambdaTarget(double lambda, vector<double> tau2, double a, double b) {
  return((-a-1)*log(lambda)-(pow(lambda,2)/2*accumulate(tau2.begin(),tau2.end(),0))-1/(b*lambda));
}

double logtau2jTarget(double tau2j, double lambda, double betaj){
  return(-log(sqrt(tau2j)) - 1.0/2.0*(1.0/tau2j*pow(betaj,2) + pow(lambda,2)*tau2j));
}

// [[Rcpp::export]]
double lambdaDraw(double current, vector<double> tau2, double a, double b){
  std::random_device rd;
  std::mt19937 mt(rd());
  std::uniform_real_distribution<double> dist(0, 1.0);
  std::normal_distribution<double> norm(0, current);
  double proposed = exp(log(current)+norm(mt));
  double logr = loglambdaTarget(proposed,tau2,1,b)-
    loglambdaTarget(current,tau2,1,b)+log(proposed)-
    log(current);
  if(log(dist(mt))<logr){
    current = proposed;
  }
  return current;
}

// [[Rcpp::export]]
vector<double> tau2Draw(vector<double> current, vector<double> beta, double lambda){
  std::random_device rd;
  std::mt19937 mt(rd());
  std::uniform_real_distribution<double> dist(0, 1.0);
  std::normal_distribution<double> norm(0, 1);
  for(int j=0;j < current.size(); j++){
    double tau2j_proposed = exp(log(current[j])+norm(mt));
    double logr =
      logtau2jTarget(tau2j_proposed, lambda, beta[j]) -
      logtau2jTarget(current[j], lambda, beta[j]) +
      log(tau2j_proposed)-log(current[j]);
    if(log(dist(mt)) < logr){
      current[j] = tau2j_proposed;
    }
  }
  return current;
}

// [[Rcpp::export]]
double sigma2Draw(const arma::vec & beta, const arma::vec & y, const arma::mat & X){
  std::random_device rd;
  std::mt19937 mt(rd());
  int n = X.n_rows;
  arma::colvec coef = arma::solve(X, y);
  arma::colvec resid = y - X*coef;
  double rss = arma::as_scalar(arma::trans(resid)*resid);
  std::gamma_distribution<double> gamma(n/2.0+0.1,1.0/(rss/2.0+0.1));
  return 1.0/gamma(mt);
}

// [[Rcpp::export]]
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

// List mcmc(double lambda, double sigma2, const arma::vec & tau2, const arma::vec & beta2){
//   
//   return List::create(Named("beta.chain") = M, Named("sigma2.chain")= m,
//                       Named("tau2.chain") = M, Named("lambda.chain") = M);
// }

