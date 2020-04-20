set.seed(1)
library(lars)
library(latex2exp)
library(mvtnorm)
library(lars)
library(glmnet)
set.seed(1)
data("diabetes")
X <- cbind(diabetes$x); y <- diabetes$y; n <- nrow(X);
X <- cbind(rep(1,n),X);p <- ncol(X)
samples <- 10; a <- b <- 0.1; delta <- 0.1
lambda2 <- 0.5
tau2 <- rep(1,p)
beta<- solve(t(X)%*%X)%*%t(X)%*%y
s <- 1
set.seed(1)
data("diabetes")
X <- cbind(diabetes$x); y <- diabetes$y; n <- nrow(X);
X <- cbind(rep(1,n),X);p <- ncol(X)
samples <- 2000;
lambda2 <- 0.5
tau2 <- rep(1,p)
beta<- solve(t(X)%*%X)%*%t(X)%*%y
s <- 1
sigma2.keep<-rep(0,samples)
beta.keep<-matrix(NA,nrow=p,ncol=samples)
a <- 0.1
b <- 100
q <- 10
r <- 10
sigma2 <- 1
sigma2.keep[1] <- sigma2
beta.keep[,1] <- solve(t(X)%*%X)%*%t(X)%*%y
for(s in 2:samples){
  lambda2.p <- runif(1,min=lambda2-0.1,max=lambda2+0.1)
  while(lambda2.p<=0){
    lambda2.p <- runif(1,min=lambda2-0.1,max=lambda2+0.1)
  }
  logr <- sum(dgamma(tau2,1,lambda2.p/2,log=T))+
    dinvgamma(lambda2.p,a,b,log=T)-
    sum(dgamma(tau2,1,lambda2/2,log=T))-
    dinvgamma(lambda2.p,a,b,log=T)
  if(log(runif(1))<logr){lambda2 <- lambda2.p}
  for(i in 1:p){
    tau2i.p <- runif(1,min=tau2[i]-0.1,max=tau2[i]+0.1)
    while(tau2i.p<=0){
      tau2i.p <- runif(1,min=tau2[i]-0.1,max=tau2[i]+0.1)
    }
    logr <- dnorm(beta[i],0,tau2i.p,log=T)+
      dgamma(tau2i.p,1,lambda2/2,log=T)-
      dnorm(beta[i],0,tau2[i],log=T)-
      dgamma(tau2[i],1,lambda2/2,log=T)
    if(log(runif(1))<logr){tau2[i] <- tau2i.p}
  }
  sigma2 <- rgamma(1,n/2+q, (r+t(y-X%*%beta)%*%(y-X%*%beta)/2))
  M <- matrix(solve(1/sigma2*t(X)%*%X+diag(1/tau2)),p,p)
  m <- M%*%t(X)%*%y/sigma2
  beta <- t(rmvnorm(n=1,mean=m,sigma=M))
  sigma2.keep[s] <- sigma2
  beta.keep[,s] <- beta
}
