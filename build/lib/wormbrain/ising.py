import numpy as np
from itertools import combinations
import matplotlib.pyplot as plt

class ising:
	def __init__(self, netsize):	#Create ising model
	
		self.size=netsize
		self.h=np.zeros(netsize)
		self.J=np.zeros((netsize,netsize))
		self.randomize_state()
		self.Beta=1
	
	def randomize_state(self):
		self.s = np.random.randint(0,2,self.size)*2-1

	
	def pdf(self):	#Get probability density function of ising model with parameters h, J
	
		self.P=np.zeros(2**self.size)
		for n in range(2**self.size):
			s=bitfield(n,self.size)*2-1
			self.P[n]=np.exp(self.Beta*(np.dot(s,self.h) + np.dot(np.dot(s,self.J),s)))
		self.Z=np.sum(self.P)
		self.P/=self.Z

	def random_wiring(self):	#Set random values for h and J
		self.h=np.random.randn(self.size)
		self.J=np.zeros((self.size,self.size))
		for i in np.arange(self.size):
			for j in np.arange(i+1,self.size):
				self.J[i,j]=np.random.randn(1)
	
	def random_rewire(self):
		if np.random.rand(1)>0.5:
			self.h[np.random.randint(self.size)]=np.random.randn(1)
		else:
			i=np.random.randint(self.size-1)
			j=np.random.randint(i,self.size)
			self.J[i,j]=np.random.randn(1)

	def independent_model(self, m):		#Set h to match an independen models with means m
		self.h=np.zeros((self.size))
		for i in range(self.size):
			self.h[i]=-0.5*np.log((1-m[i])/(1+m[i]))
		self.J=np.zeros((self.size,self.size))
		
	def observables(self):		#Get mean and correlations from probability density function
		self.pdf()
		self.m=np.zeros((self.size))
		self.C=np.zeros((self.size,self.size))
		for n in range(2**self.size):
			s=bitfield(n,self.size)*2-1
			self.m+=self.P[n]*s
			for i in range(self.size):
#				self.m[i]+=self.P[n]*s[i]
				self.C[i,i+1:]+=self.P[n]*s[i]*s[i+1:]
#				for j in np.arange(i+1,self.size):
#					self.C[i,j]+=self.P[n]*s[i]*s[j]
		for i in range(self.size):
			self.C[i,i+1:]-=self.m[i]*self.m[i+1:]
#			for j in np.arange(i+1,self.size):
#				self.C[i,j]-=self.m[i]*self.m[j]
				
	def inverse_exact(self,m1,C1,error,mode='gradient-descent'):	#Solve exact inverse ising problem with gradient descent
		u=0.04
		count=0
		self.independent_model(m1)
		
		self.observables()
		fit = max (np.max(np.abs(self.m-m1)),np.max(np.abs(self.C-C1)))
		fmin=fit
		
		
		while fit>error:
			
			if mode=='gradient-descent':
			
				dh=u*(m1-self.m)
				self.h+=dh
				dJ=u*(C1-self.C)
				self.J+=dJ
			elif mode=='coordinated-descent':
			
				beta=fit*0.001
#				beta=error**2
				
				def compF(d,p,ql,l,beta):
					return -d*p +np.log(np.exp(-d) + (np.exp(d)-np.exp(-d))*(ql+1)*0.5)+beta*(np.abs(l+d)-np.abs(l))

				def Fmin(p,ql,l,beta):
					D=[]
					for B in [1,-1]:
						nden=(1+ql)*(1-p+B*beta)
						if not nden==0:
							nnum=(1-ql)*(1+p-B*beta)
							if nnum/nden>0:
								D1=0.5*np.log(nnum/nden)
								if B*(l+D1)>0:
									D+=[D1]
					if len(D)==1:
						return D[0]
					else:
#						print compF(-l,p,ql,l,beta)
						print ('error',len(D))
						return None
#						
#						plt.figure()
##						print l
##						print ql,p
#						d=np.arange(-2,2,0.001)
#						plt.plot(d,compF(d,p,ql,l,beta))
#						plt.show()
						
#						exit(0)

				inds=[]
				p=[]
				ql=[]
				l=[]
				for i in range(self.size):
					inds+=[i]
					p+=[m1[i]]
					ql+=[self.m[i]]
					l+=[self.h[i]]
				for i in range(self.size):
					for j in np.arange(i+1,self.size):
						inds+=[(i,j)]
						p+=[C1[i,j]]
						ql+=[self.C[i,j]]
						l+=[self.J[i,j]]
				N=len(inds)
				F=np.zeros(N)
				d=np.zeros(N)
				for i in range(len(inds)):
					d[i]=Fmin(p[i],ql[i],l[i],beta)
					if np.isnan(d[i]):
						F[i]=1E10
						d[i]=u*(p[i]-ql[i])
					else:
						F[i]=compF(d[i],p[i],ql[i],l[i],beta)

				ind=np.argmin(F)
				D=d[ind]

				if ind<self.size:
					self.h[inds[ind]]+=d[ind]*1
				else:
					self.J[inds[ind]]+=d[ind]*1

#			

			fit = max (np.max(np.abs(self.m-m1)),np.max(np.abs(self.C-C1)))
			count+=1
			if count%1==0:
				print (self.size,count,fit)
			self.observables()
			
			
		return fit
				
	def inverseMC(self,m1,C1,error):	#Solve inverse ising problem using Monte Carlo Samples

		u=0.1
		samples=200
		nT=40

		fit=1E10
		fmin=fit
		fitcount=0
		self.independent_model(m1)
		PS=[]
		fits=[]
		for i in range(nT):
			PS+=[self.MCsamples(samples)]
			fits+=[fit]

		count=0
		while fit>error:
			del PS[0]
			PS+=[self.MCsamples(samples)]
			Ps=PS[0]
			for i in np.arange(1,nT):
				Ps.update(PS[i])

			ns=Ps.keys()
			Pmc=Ps.values()
			Pmc/=np.sum(Pmc)
			self.observables_sample(ns,Pmc)
	
			dh=u*(m1-self.m)
			self.h+=dh
			dJ=u*(C1-self.C)
			self.J+=dJ
			fmin=np.min(fits)
			del fits[0]
			fit = max (np.max(np.abs(self.m-m1)),np.max(np.abs(self.C[np.abs(self.C)>0]-C1[np.abs(self.C)>0])))
			fits+=[fit]
			if fit/fmin<1:
	#			fmin=fit
				fitcount=0
			else:
				fitcount+=1
				if fitcount>nT*2:
					if len(Ps)/2.0**self.size<1:
						samples+=samples/2
					fitcount=0
			if count%10==0:
				print (self.size,count,len(Ps)/2.0**self.size,samples,fit)
			count+=1
	
		return fit
	
	
	def inverse_sampler(self,m1,C1,error):	#Solve inverse ising problem using Monte Carlo Samples

		u=0.01

		fit=1E10
		fmin=fit
		fitcount=0
		T=10000
		nT=40
		count=0
		while fit>error:
			if count%nT==0:
				samples=self.generateMCsample(T)
				h0=self.h.copy()
				J0=self.J.copy()
				self.observables_sample(samples)
				fit = max (np.max(np.abs(self.m-m1)),np.max(np.abs(self.C-C1)))
			self.observables_recycled_samples(samples,h0,J0)
			dh=u*(m1-self.m)
			self.h+=dh
			dJ=u*(C1-self.C)
			self.J+=dJ
			fit1 = max (np.max(np.abs(self.m-m1)),np.max(np.abs(self.C-C1)))

			if count%10==0:
				print (self.size,count,fit,fit1)
			count+=1
	
		return fit
				
	def MCsamples(self,samples):	#Generate a series of Monte Carlo samples
		self.randomize_state()
		# Main    simulation loop:
		P={}
		for t in range(samples):
			self.MetropolisStep()
			n=bool2int((self.s+1)/2)
			if n<0:
				print( n)
				print ((self.s+1)/2)
			P[n]=np.exp((np.dot(self.s,self.h) + np.dot(np.dot(self.s,self.J),self.s)))
		return P
		
	def generateMCsample(self,T):	#Generate a series of Monte Carlo samples
		self.randomize_state()
		# Main simulation loop:
		samples=[]
		for t in range(T):
			self.MetropolisStep()
#			self.SequentialGlauberStep()
			n=bool2int((self.s+1)/2)
			samples+=[n]
		return samples

	def observables_sample(self, samples):	#Get mean and correlations from system states sample
		
		ns,P=np.unique(samples,return_counts=True)
		P=P.astype(float)
		P/=np.sum(P)
		self.m=np.zeros((self.size))
		self.C=np.zeros((self.size,self.size))
		for ind,n in enumerate(ns):
			s=bitfield(n,self.size)*2-1
			for i in range(self.size):
				self.m[i]+=P[ind]*s[i]
				for j in np.arange(i+1,self.size):
					self.C[i,j]+=P[ind]*s[i]*s[j]
		for i in range(self.size):
			for j in np.arange(i+1,self.size):
				self.C[i,j]-=self.m[i]*self.m[j]
				
					
	def observables_recycled_samples(self, samples,h0,J0):	#Get 'recycled' mean and correlations from system states sample and new values of h and J

		ns,P=np.unique(samples,return_counts=True)
		P=P.astype(float)
		P/=np.sum(P)
		self.m=np.zeros((self.size))
		self.C=np.zeros((self.size,self.size))
		for ind,n in enumerate(ns):
			s=bitfield(n,self.size)*2-1
			Pdiff=np.exp((np.dot(s,(self.h-h0)) + np.dot(np.dot(s,(self.J-J0)),s)))
			P1=P[ind]*Pdiff
			for i in range(self.size):
				self.m[i]+=P1*s[i]
				for j in np.arange(i+1,self.size):
					self.C[i,j]+=P1*s[i]*s[j]
		for i in range(self.size):
			for j in np.arange(i+1,self.size):
				self.C[i,j]-=self.m[i]*self.m[j]
			
	def MetropolisStep(self,i=None):	    #Execute step of Metropolis algorithm
		if i is None:
			i = np.random.randint(self.size)
		eDiff = self.deltaE(i)
		if eDiff <= 0 or np.random.rand(1) < np.exp(-self.Beta*eDiff):    # Metropolis
			self.s[i] = -self.s[i]
			
	def MetropolisStepT0(self,i=None):	    #Execute step of Metropolis algorithm with zero temperature (deterministic)
		if i is None:
			i = np.random.randint(self.size)
		eDiff = self.deltaE(i)
		if eDiff <= 0:
			self.s[i] = -self.s[i]
			
	def GlauberStep(self,i=None):			#Execute step of Glauber algorithm
		if i is None:
			i = np.random.randint(self.size)
		eDiff = self.deltaE(i)
		if np.random.rand(1) < 1.0/(1.0+np.exp(eDiff)):    # Glauber
			self.s[i] = -self.s[i]
			
	def SequentialGlauberStep(self):
		for i in np.random.permutation(self.size):
			self.GlauberStep(i)


	def deltaE(self,i):		#Compute energy difference between two states with a flip of spin i
		return 2*(self.s[i]*self.h[i] + np.sum(self.s[i]*(self.J[i,:]*self.s)+self.s[i]*(self.J[:,i]*self.s)))
 

	def metastable_states(self):	#Find the metastable states of the system
		self.pdf()
		self.ms=[]
		Pms=[]
		for n in range(2**self.size):
			m=1
			s=bitfield(n,self.size)
			for i in range(self.size):
				s1=s.copy()
				s1[i]=1-s1[i]
				n1=bool2int(s1)
				if self.P[n]<self.P[n1]:
					m=0
					break
			if m==1:
				self.ms+=[n]
				Pms+=[self.P[n]]
		return Pms
		
	def get_valley(self,s):		#Find an attractor "valley" starting from state s

		n=bool2int((s+1)/2)
		self.s=s.copy()
		while not n in self.ms:	
			self.MetropolisStepT0()
			n=bool2int((self.s+1)/2)
		ind=self.ms.index(n)
		valley=ind
#		print ind,n,ms
		return valley
		
	def energy(self):	#Compute energy function
		self.pdf()
		self.E=np.zeros(2**self.size)
		for n in range(2**self.size):
			s=bitfield(n,self.size)*2-1
			self.E[n]=-(np.dot(s,self.h) + np.dot(np.dot(s,self.J),s))
		self.Em=np.sum(self.P*self.E)

	def HeatCapacity(self):	#Compute energy function
		self.energy()
		self.HC=self.Beta**2*(np.sum(self.P*self.E**2)-np.sum(self.P*self.E)**2)

	
def bool2int(x):				#Transform bool array into positive integer
    y = 0
    for i,j in enumerate(np.array(x)[::-1]):
#        y += j<<i
        y += j*2**i
    return y
    
def bitfield(n,size):			#Transform positive integer into bit array
    x = [int(x) for x in bin(n)[2:]]
    x = [0]*(size-len(x)) + x
    return np.array(x)

def subPDF(P,rng):
	subsize=len(rng)
	Ps=np.zeros(2**subsize)
	size=int(np.log2(len(P)))
	for n in range(len(P)):
		s=bitfield(n,size)
		Ps[bool2int(s[rng])]+=P[n]
	return Ps
	
def Entropy(P):
	E=0.0
	for n in range(len(P)):
		if P[n]>0:
			E+=-P[n]*np.log(P[n])
	return E
	

def MI(Pxy, rngx, rngy):
	size=int(np.log2(len(Pxy)))
	Px=subPDF(Pxy,rngx)
	Py=subPDF(Pxy,rngy)
	I=0.0
	for n in range(len(Pxy)):
		s=bitfield(n,size)
		if Pxy[n]>0:
			I+=Pxy[n]*np.log(Pxy[n]/(Px[bool2int(s[rngx])]*Py[bool2int(s[rngy])]))
	return I
	
def TSE(P):
	size=int(np.log2(len(P)))
	C=0
	for npart in np.arange(1,0.5+size/2.0).astype(int):	
		bipartitions = list(combinations(range(size),npart))
		for bp in bipartitions:
			bp1=list(bp)
			bp2=list(set(range(size)) - set(bp))
			C+=MI(P, bp1, bp2)/float(len(bipartitions))
	return C
	
def KL(P,Q):
	D=0
	for i in range(len(P)):
		D+=P[i]*np.log(P[i]/Q[i])
	return D
    
def JSD(P,Q):
	return 0.5*(KL(P,Q)+KL(Q,P))
