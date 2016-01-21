# Simple 1D GP regression example
import numpy as np
import scipy.optimize as op
import matplotlib.pyplot as plt
np.random.seed(0)

# Define squared distance calculation function
def squaredDistance(A,B):
	A = np.reshape(A,(len(A),1))
	B = np.reshape(B,(len(B),1))
	A2_sum = A*A
	B2_sum = B*B
	AB = 2*np.dot(A,B.T)
	A2 = np.tile(A2_sum,(1,np.size(B2_sum,axis=0)))
	B2 = np.tile(B2_sum.T,(np.size(A2_sum,axis=0),1))
	sqDist = A2 + B2 - AB
	return sqDist

# Define GP class
class gaussianProcess(object):
	def __init__(self,log_hyp,mean_hyp,like_hyp,covFunName,meanFunName,likeFunName,
				 trainInput,trainTarget):
		self.log_hyp = log_hyp
		self.mean_hyp = mean_hyp
		self.like_hyp = like_hyp
		self.covFunName = covFunName
		self.meanFunName = meanFunName
		self.likeFunName = likeFunName
		self.trainInput = trainInput
		self.trainTarget = trainTarget
		# Can you pass class handles instead of doing it the dumb way below?
		if covFunName == "SE":
                    self.covFun = squaredExponential(log_hyp,trainInput)
		else:
                    self.covFun = []
		
		if meanFunName == "zero":
                    self.meanFun = meanFun(mean_hyp)
		else:
                    self.meanFun = []
		
		
		if likeFunName == "zero":
			self.likeFun = likeFun(like_hyp)
		else:
			self.likeFun = []
		
	# Define GP prediction function
	def computePrediction(self,testInput):
		Kxz = self.covFun.computeKxzMatrix(testInput)
		Kxx = self.covFun.computeKxxMatrix()
		iKxx = np.linalg.inv(Kxx)
		Kzx = Kxz.T
		K_diag = np.diagonal(np.dot(np.dot(Kzx,iKxx),Kxz))
		K_noise = self.covFun.sf2*np.ones(np.size(testInput,axis=0))
		fz = np.dot(np.dot(Kzx,iKxx),self.trainTarget.T)
		cov_fz = K_noise - K_diag
		return fz, cov_fz
	
	# Define GP negative log marginal likelihood function
	def computeLikelihood(self,hyp):
		n = np.size(self.trainInput,axis=0)
		covSE = squaredExponential(hyp,self.trainInput)
		Kxx = covSE.computeKxxMatrix()
		m = self.meanFun.y
		L = np.linalg.cholesky(Kxx)
		iKxx = np.linalg.solve(L.T,np.linalg.solve(L,np.eye(n)))
		y = np.reshape(self.trainTarget,(len(self.trainTarget),1))
		err_y = np.dot(np.dot((y-m).T,iKxx),(y-m))/2
		det_Kxx = np.sum(np.log(np.diag(L)))
		occams_razor = n*np.log(2*np.pi)/2
		nlml = err_y + det_Kxx + occams_razor
		return nlml

# Define GP mean function class
class meanFun(object):
	def __init__(self,x):
		self.x = x
		self.y = np.zeros_like(x)

# Define GP likelihood function class
class likeFun(object):
	def __init__(self,x):
		self.x = x
		self.y = np.exp(2*x)	

# Define covariance function class
class covarianceFunction(object):
	def __init__(self,logHyp,x):
		self.logHyp = logHyp 			# log hyperparameters
		self.x = x						# training inputs

# Define squared exponential covariance function
class squaredExponential(covarianceFunction):
	def __init__(self,logHyp,x):
		covarianceFunction.__init__(self,logHyp,x)
		self.hyp = np.exp(self.logHyp)	# hyperparameters
		n = len(self.hyp)
		self.M = self.hyp[:n-2]			# length scales
		self.sf2 = self.hyp[n-2]**2		# squared exponential variance
		self.sn2 = self.hyp[n-1]**2		# noise variance
	
	def computeKxxMatrix(self):
		scaledX = self.x/self.M
		sqDist = squaredDistance(scaledX,scaledX)
		Kxx = self.sn2*np.eye(np.size(self.x,axis=0))+self.sf2*np.exp(-0.5*sqDist)
		return Kxx
	
	def computeKxzMatrix(self,z):
		scaledX = self.x/self.M
		scaledZ = z/self.M
		sqDist = squaredDistance(scaledX,scaledZ)
		Kxz = self.sf2*np.exp(-0.5*sqDist)
		return Kxz

# Main program
# Plot true function
x_plot = np.arange(0,1,0.01,float)
y_plot = trueFunction(x_plot)
plt.figure()
plt.plot(x_plot,y_plot,'k-')

# Training data
x_train = np.random.random(20)
y_train = obsFunction(x_train)
plt.plot(x_train,y_train,'rx')

# Test data
x_test = np.arange(0,1,0.01,float)

# GP hyperparameters
# note: using log scaling to aid learning hyperparameters with varied magnitudes
log_hyp = np.log([1,1,0.1])
mean_hyp = 0
like_hyp = 0

# Initialise GP for hyperparameter training
initGP = gaussianProcess(log_hyp,mean_hyp,like_hyp,"SE","zero","zero",x_train,y_train)

# Run optimisation routine to learn hyperparameters
opt_log_hyp = op.fmin(initGP.computeLikelihood,log_hyp)

# Learnt GP with optimised hyperparameters
optGP = gaussianProcess(opt_log_hyp,mean_hyp,like_hyp,"SE","zero","zero",x_train,y_train)
y_test,cov_y = optGP.computePrediction(x_test)

# Plot true and modelled functions
plt.plot(x_test,y_test,'b-')
plt.plot(x_test,y_test+np.sqrt(cov_y),'g--')
plt.plot(x_test,y_test-np.sqrt(cov_y),'g--')
plt.show()
