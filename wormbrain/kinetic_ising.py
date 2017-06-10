import numpy as np
import matplotlib.pyplot as plt

from itertools import combinations
from scipy.special import erfinv
from numba import jit
import numexpr as ne

    
def bool2int(x):                #Transform bool array into positive integer
    y = 0
    for i,j in enumerate(np.array(x)[::-1]):
#        y += j<<i
        y += j*2**i
    return y
    
def bitfield(n,size):            #Transform positive integer into bit array
    x = [int(x) for x in bin(int(n))[2:]]
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

    
def PCA(h,J):
    size=len(h)
    P=get_PDF(h,J,size)
    m,C=observables(P,size)
    C=0.5*(C+np.transpose(C))
    w,v = np.linalg.eig(C)
    return w,v

class ising:
    def __init__(self, netsize):    #Create ising model
    
        self.size=netsize
        self.h=np.zeros(netsize)
        self.J=np.zeros((netsize,netsize))
        self.T=1
        self.randomize_state()
        self.convergencias_malas = 0
        self.divergencias = 0
        self.esperas = 100

    def color_bar(self, data):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ticks_at = [-abs(data).max(), 0, abs(data).max()]
        cax = ax.imshow(data, interpolation='nearest', 
                        origin='lower', extent=[0.0, 0.1, 0.0, 0.1],
                        vmin=ticks_at[0], vmax=ticks_at[-1])
        cbar = fig.colorbar(cax,ticks=ticks_at,format='%1.2g')
        return cbar
    
    def randomize_state(self):
        self.s = np.random.randint(0,2,self.size)*2-1        
        
    def pdfMC(self,T):    #Get mean and correlations from Monte Carlo simulation of the kinetic ising model
        self.P=np.zeros(2**self.size)
        self.randomize_state()
        for t in range(T):
            self.GlauberStep()
            n=bool2int((self.s+1)/2)
            self.P[n]+=1.0/float(T)

    def random_wiring(self):    #Set random values for h and J
        self.h=np.random.rand(self.size)*2-1
        self.J=np.random.randn(self.size,self.size)
                
            
    def GlauberStep(self):
        s=self.s.copy()
        for i in range(self.size):
            eDiff = self.deltaE(s,i)
            if np.random.rand() < 1.0/(1.0+np.exp(eDiff/self.T)):    # Glauber
                self.s[i] = -self.s[i]
        
    def deltaE(self,s,i=None, parallel = False):
        if i is None:
            return 2*(s*self.h + s*np.dot(s,self.J) )
        else:
            return 2*(s[i]*self.h[i] + np.sum(s[i]*(self.J[:,i]*s)))
    
    def H(self,i=None):
        if i is None:
            return self.h + np.dot(self.s, self.J)
        else:
            return self.h[i] + np.sum(self.J[:,i] * self.s)

    def observables_sample(self,sample):    #Get mean and correlations from Monte Carlo simulation of the kinetic ising model
        self.m=np.zeros(self.size)
        self.D=np.zeros((self.size,self.size))
        ns,P=np.unique(sample,return_counts=True)
        P=P.astype(float)
        P/=np.sum(P)
        for ind,n in enumerate(ns):
            s=bitfield(n,self.size)*2-1 #Pasar el estado n a bits
            eDiff = self.deltaE(s) #Calcular la energia de ese estado
            pflip=1.0/(1.0+np.exp((1/self.T)*eDiff)) #Calcular la probabilidad de cambiar de signo
            self.m+= (s*(1-2*pflip))*P[ind]
            d1, d2 = np.meshgrid(s*(1-2*pflip),s)
            self.D+= d1*d2*P[ind]

        for i in range(self.size):
            for j in range(self.size):
                self.D[i,j]-=self.m[i]*self.m[j]
                

    def generate_sample(self,T = None, state = None, booleans = False):    #Generate a series of Monte Carlo samples
        '''
        Genera una muestra de estados de Ising.
        
        state -- si no se facilita un estado inicial, se generara uno aleatorio.
        '''
        if T is None:
            minimum = 1000
            maximum = 1000000
            T = 2**self.size
            if T > maximum:
                T = maximum
            if T < minimum:
                T = minimum
                
        if state is None:
                self.randomize_state()
        else:
                self.s = state
                
        samples=[]
        result = []
        for t in range(T):
            self.GlauberStep()
            n=bool2int((self.s+1)/2)
            samples+=[n]
        
        if booleans:
            for i in range(T):
                result.append(bitfield(samples[i], self.size).tolist())
            
            return np.array(result)
        else:        
            return samples
        
        
    def independent_model(self, m):        #Set h to match an independen models with means m
        '''
        Hace que el sistema Ising se ajuste a un modelo independiente de media m.
        (Los J quedan todos a 0)
        '''
        self.h=np.zeros((self.size))
        for i in range(self.size):
            self.h[i]=-0.5*np.log((1-m[i])/(1+m[i]))
        self.J=np.zeros((self.size,self.size))
        
    def diverge(self, errores, u):
       '''
        Comprueba si los errores han divergido/estan divergiendo.
       '''
       proporciones = np.zeros(len(errores)-1)
       aumentos = np.zeros(len(errores)-1)
       for i in np.arange(1,len(errores)):
          proporciones[i-1]= errores[i]/errores[i-1]
          aumentos[i-1] = errores[i] < errores[i-1]
       proporciones = abs(proporciones - 1)
       if (abs(sum(aumentos)) > len(aumentos)*0.4) and (abs(sum(aumentos)) < len(aumentos)*0.6) and (np.sum(proporciones)>u):            
            return True
      
       return False
      
    def converge_mal(self,errores, u):
       '''
       Devuelve true si solo si los errores pasados por paramtro convergen a un
       valor. (Normalmente, este valor no es el que buscamos)
       '''
       proporciones = np.zeros(len(errores)-1)
       aumentos = np.zeros(len(errores)-1)
       for i in np.arange(1,len(errores)):
          proporciones[i-1]= errores[i]/errores[i-1]
          aumentos[i-1] = errores[i] < errores[i-1]
       proporciones = abs(proporciones - 1)
       if (abs(sum(aumentos==True)-sum(aumentos==False)) < 2) and (np.mean(proporciones)<u):
            if self.convergencias_malas < 3:
                self.convergencias_malas += 1
                return False 
            else:
                self.convergencias_malas = -10
                return True
            
       self.convergencias_malas = 0
       return False
        
    #@jit
    def inverse(self,m1,D1, error,sample, maximum_error = True, u = 0.01, verboso = True, max_iteration = np.inf, parallel = False):
        '''
        Entrena el sistema de Ising utilizando descenso de gradiente.
        
        parallel -- utiliza numexpr para acelerar algunas funciones (experimental).
            Solo util para sistemas muy grandes (donde las matrices de pesos y medias son muy costosas de operar.)
            Recomendado no utilizarlas a menos que uno sepa lo que hace.
        '''
        count=0
        self.observables_sample(sample)
        errores = np.zeros(10)
        errores_divergencia = np.zeros(100)
        hold = False

        if (maximum_error):
               fit = max (np.max(np.abs(self.m-m1)),np.max(np.abs(self.D-D1)))
        else:
               fit = max (np.mean(np.abs(self.m-m1)),np.mean(np.abs(self.D-D1))) 
        
        while (fit>error) and (max_iteration>count):
            self.observables_sample(sample)
            
            if parallel:
                m = self.m
                D = np.array(self.D)
                h = self.h
                J = self.J
                dh = ne.evaluate("u*(m1-m)")
                dJ = ne.evaluate("u*(D1-D)")
                self.h = ne.evaluate("h + dh")
                self.J = ne.evaluate("J+dJ")
            else:
                dh=u*(m1-self.m)
                dJ=u*(D1-self.D)
                self.h+=dh
                self.J+= dJ
            
            
            if (maximum_error):
               fit = max (np.max(np.abs(self.m-m1)),np.max(np.abs(self.D-D1)))
            else:
               fit = max (np.mean(np.abs(self.m-m1)),np.mean(np.abs(self.D-D1)))
               
            errores[count%10-1] = fit
            errores_divergencia[count%100-1] = fit
               
            if (count>0) and (count%10==0):
               if (not hold): 
                   if (self.esperas == 100):
                       if (self.diverge(errores_divergencia, u)) :
                           print("Reajustado el factor de aprendizaje por divergencia (Esperamos a que deje de crecer)")   
                           hold = True
                           self.esperas = 0
                   else:
                       self.esperas = min(self.esperas + 1, 100)
                   if (self.converge_mal(errores, u)):
                        print("Rajustado el factor de aprendizaje por convergencia mala")
                        u = u / 10.0
               else:
                   if (errores[-1] < errores[-2]):
                       u = u / 10.0
                       hold = False
              
               if verboso:
                  print(self.size,count,fit,u)
                  
            count+=1
        
        return fit
        

