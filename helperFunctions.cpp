#include <Rcpp.h>
#include <cmath>
#include <math.h>
#include <random>
using namespace Rcpp;
using namespace std;

double loglambdaTarget(double lambda, vector<double> tau2, double a, double b) {
  return((-a-1)*log(lambda)-(pow(lambda,2)/2*accumulate(tau2.begin(),tau2.end(),0))-1/(b*lambda));
}

// [[Rcpp::export]]
double logtau2jTarget(double tau2j, double lambda, double betaj){
  return(-log(sqrt(tau2j)) - 1.0/2.0*(1.0/tau2j*pow(betaj,2) + pow(lambda,2)*tau2j));
}

// [[Rcpp::export]]
double lambdaDraw(double proposed, double current, vector<double> tau2, double a, double b){
  double accept = current;
  double logr = loglambdaTarget(proposed,tau2,1,b)-
    loglambdaTarget(current,tau2,1,b)+log(proposed)-
    log(current);
  std::random_device rd;
  std::mt19937 mt(rd());
  std::uniform_real_distribution<double> dist(0, 1.0);
  if(log(dist(mt))<logr){
    accept = proposed;
  }
  return accept;
}
// [[Rcpp::export]]
vector<double> tau2Draw(vector<double> current, vector<double> beta, double lambda){
  std::random_device rd;
  std::mt19937 mt(rd());
  std::uniform_real_distribution<double> dist(0, 1.0);
  std::normal_distribution<double> norm(0, 1.0);
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
