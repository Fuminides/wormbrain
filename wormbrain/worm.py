import xlrd
import csv

import numpy as np
import matplotlib.pyplot as plt
from . import analyze_model as am

from scipy.optimize import leastsq

##################################################
# Funciones
##################################################

def quick_load(gusano, tipo_compresion, umbral_usado):
    '''
    Carga umbralizada y comprimida la actividad neuronal del gusano especificada.
    
    gusano -- numero del gusano a cargar
    tipo_compresion, umbral_usado -- consultar funciones de analyze_model
    '''
    (neural_activation,behavior)=get_neural_activation(gusano)
    neural_activation = am.compresion(neural_activation, behavior, tipo_compresion)
    
    return am.umbralizar(neural_activation,umbral_usado)

def get_real_adjancency():
    '''
    Carga el fichero de adyacencias reales de las neuronas un C.ellegans.
    Devuelve una matriz.
    '''
    lector = csv.reader(open("./files/celegans277matrix.csv", 'r'), delimiter=',')
    resultado = np.zeros([277,277])
    i = 0
    for row in lector:
        resultado[i] = row
        i += 1
        
    return resultado

def asignar_conexion(neurona1, neurona2, neuronas, sinapsis, indice, conexiones):
    if not (neurona1 in neuronas):
        neuronas[neurona1] = indice
        indice += 1
    
    if not (neurona2 in neuronas):
        neuronas[neurona2] = indice
        indice += 1
    
    ind1 = neuronas[neurona1]
    ind2 = neuronas[neurona2]
    
    if sinapsis == 'Gap_junction':
        conexiones[ind1,ind2] = True
        conexiones[ind2,ind1] = True
    elif sinapsis == 'Send':
        conexiones[ind1,ind2] = True
    elif sinapsis == 'Send_joint':
        conexiones[ind1,ind2] = True
    elif sinapsis == 'Receive':
        conexiones[ind2,ind1] = True
    elif sinapsis == 'Receive_joint':
        conexiones[ind2,ind1] = True
        
    return indice
    
def load_real_connections():
    '''
    Devuelve una matriz de conexion con las conexiones reales del gusano.
    '''
    lectura_fichero = open("./files/neurodata.txt")
    neurons = {}
    indice = 0
    resultado = np.zeros([200, 200]) #Tamano calculado a partir del dataset
    for line in lectura_fichero:
        claves = str.split(lectura_fichero.readline(), '\t')
        neurona1 = claves[0]
        neurona2 = claves[1]
        sinapsis = claves[2]
        indice = asignar_conexion(neurona1, neurona2, neurons, sinapsis, indice, resultado)
        
    return resultado
    
def get_neural_activation(sheet_index=0, filtro = True):
   
	xl_workbook = xlrd.open_workbook('files/pnas.1507110112.sd01.xls')
#	print xl_workbook.sheet_names()
	xl_sheet = xl_workbook.sheet_by_index(sheet_index)	#Select sheet [0-4]
#	print ('Sheet name: %s' % xl_sheet.name)

	Nrows = xl_sheet.nrows
	Ncols = xl_sheet.ncols

	X=np.zeros((Nrows-1,Ncols))

	for j in range(Ncols):

		col = xl_sheet.col(j)
		for i in np.arange(1,Nrows):
	
			if str(col[i])[:6]=='number':
				X[i-1,j]=float(str(col[i])[7:])

	neural_activation=X[:,4:]
	behavior=X[:,3]
	if  filtro:
		neural_activation = np.load("./files/worm"+str(sheet_index)+"-filtered.npy")
    
	return neural_activation,behavior

def get_center_position(sheet_index=0):

	xl_workbook = xlrd.open_workbook('files/pnas.1507110112.sd01.xls')
#	print xl_workbook.sheet_names()
	xl_sheet = xl_workbook.sheet_by_index(sheet_index)	#Select sheet [0-4]
#	print ('Sheet name: %s' % xl_sheet.name)

	Nrows = xl_sheet.nrows
	Ncols = xl_sheet.ncols

	X=np.zeros((Nrows-1,Ncols))

	for j in range(Ncols):

		col = xl_sheet.col(j)
		for i in np.arange(1,Nrows):
	
			if str(col[i])[:6]=='number':
				X[i-1,j]=float(str(col[i])[7:])

	t=X[:,0]
	x=X[:,1]
	y=X[:,2]
	return t,x,y


def ls_sine(data,f):


	N = len(data)
	t = np.linspace(0, f*np.pi, N)

	guess_mean = np.mean(data)
	guess_std = 3*np.std(data)/(2**0.5)
	guess_phase = 0

	# we'll use this to plot our first estimate. This might already be good enough for you
	data_first_guess = guess_std*np.sin(t+guess_phase) + guess_mean

	# Define the function to optimize, in this case, we want to minimize the difference
	# between the actual data and our "guessed" parameters
	optimize_func = lambda x: x[0]*np.sin(t+x[1]) + x[2] - data
	est_std, est_phase, est_mean = leastsq(optimize_func, [guess_std, guess_phase, guess_mean])[0]

	# recreate the fitted curve using the optimized parameters
	data_fit = est_std*np.sin(t+est_phase) + est_mean
	return data_fit,est_phase,est_mean
	
def get_positions(sheet_index=0):

	ind0 = 2+ sheet_index*3 
	ind = ind0

	M=102
	t=100
	
	with open('files/pnas.1507110112.sd'+("%02d" % (ind0,))+'.csv', 'rt') as csvfile:
		reader = csv.reader(csvfile, delimiter=',', quotechar='|')
		for row in reader:
			N=len(row)
			break
			
	x=np.zeros((M,N))
	y=np.zeros((M,N))
	t=np.zeros(N)
	
	count =0
	with open('files/pnas.1507110112.sd'+("%02d" % (ind0,))+'.csv', 'rt') as csvfile:
		reader = csv.reader(csvfile, delimiter=',', quotechar='|')
		for row in reader:
			x[count,:] = np.array(row,float)
			count+=1
			
	count =0
	with open('files/pnas.1507110112.sd'+("%02d" % (ind0+1,))+'.csv', 'rt') as csvfile:
		reader = csv.reader(csvfile, delimiter=',', quotechar='|')
		for row in reader:
			y[count,:] = np.array(row,float)
			count+=1
			
	count =0
	with open('files/pnas.1507110112.sd'+("%02d" % (ind0+2,))+'.csv', 'rt') as csvfile:
		reader = csv.reader(csvfile, delimiter=',', quotechar='|')
		for row in reader:
			t = np.array(row,float)
			count+=1
	t = t-t[0]	
	


	return t,x,y

def get_movement(sheet_index=0):
	
	M=102
	t=100
	t,x,y=get_positions(sheet_index)

	(t0,x0,y0)=get_center_position(sheet_index)
	N0=len(t0)
	
	x1=np.zeros((M,N0))
	y1=np.zeros((M,N0))
	for i in range(M):
		x1[i,:]=np.interp(t0, t, x[i,:])
		y1[i,:]=np.interp(t0, t, y[i,:])
	
	x=x1+x0*1000
	y=y1+y0*1000

	t=t0
	N=N0

	dx = np.diff(x, n=1, axis=0)
	dy = np.diff(y, n=1, axis=0)
	angles = np.arctan2(dy, dx)
	forward=np.zeros(N-1)
	turning=np.zeros(N-1)
#	va=np.zeros(N-1)
#	vt=np.zeros(N-1)
#	for i in range(N-1):
#		angles[:,i]=np.unwrap(-angles[:,i])
#		turning[i] = np.sin(np.mean(angles[0:10,i+1]-angles[0:10,i]))
#		F=np.fft.fft(angles[:,i]-np.mean(angles[:,i]))
#		S=np.abs(F[0:M/2])**2*np.arange(M/2)
#		f=np.argmax(S*(np.arange(M/2)>3))	
#		(sinefit,est_phase,est_mean) = ls_sine(angles[:,i]-np.mean(angles[:,i]),f)
#		(sinefit1,est_phase1,est_mean1) = ls_sine(angles[:,i+1]-np.mean(angles[:,i+1]),f)
#		forward[i]=np.sin(est_phase-est_phase1)
#		
#		va[i]=np.mean(angles[0:10,i+1]-angles[0:10,i])
	
	p = np.arctan2(y[0,:]-y[10,:], x[0,:]-x[10,:])	#Get orientation of the head of the worm
	vx = np.diff(np.mean(x[:,:],axis=0))
	vy = np.diff(np.mean(y[:,:],axis=0))
	w = np.arctan2(vy, vx)							#Get orientation of the head's velocity
	dp=np.diff(np.unwrap(p))
	dw=w-p[0:-1]
	vt=np.sqrt(vx**2+vy**2)*np.sign(np.cos(dw))
	va=dp

#	plt.figure()
#	plt.plot(np.sqrt((y[0,:]-y[10,:])**2 + (x[0,:]-x[10,:])**2))
#	
#	plt.figure()
#	plt.plot(Lhead)
#	
	R=0
	Mr=vt
	Ml=vt
	while np.mean((Mr>0)==(Ml>0))>0.8:
		R+=1
		Mr=vt+R*va
		Ml=vt-R*va
#		print R,np.mean((Mr>0)==(Ml>0))
#	print np.mean((Mr>0)==(Ml>0))
#	print np.mean((Mr>0))
#	print np.mean((Ml>0))
	
#	plt.figure()
#	plt.plot(va)
#	plt.figure()
#	plt.plot(vt)
#	plt.figure()
#	plt.plot(Mr)
#	plt.plot(Ml)

	return Ml,Mr
	
	
#	
#	import matplotlib.pyplot as plt
#	plt.figure()
#	i=np.random.randint(N-1)
#	plt.plot(angles[:,i]-np.mean(angles[:,i]))
#	plt.plot(angles[:,i+1]-np.mean(angles[:,i+1]))
#	plt.figure()
#	plt.plot(x[:,i],y[:,i])
#	plt.plot(x[0,i],y[0,i],'o')
#	plt.plot(x[:,i+1],y[:,i+1])
#	plt.plot(x[0,i+1],y[0,i+1],'o')

#	print np.mean(angles[0:10,i+1]-angles[0:10,i])
#	

#	
#	
#	F=np.fft.fft(angles[:,i]-np.mean(angles[:,i]))
#	plt.figure()
#	plt.plot(np.abs(F[0:M/2])**2*np.arange(M/2))
#	plt.plot(np.abs(F[0:M/2])**2)
#	print np.argmax(np.abs(F))
#	S=np.abs(F[0:M/2])**2*np.arange(M/2)
#	f=np.argmax(S*(np.arange(M/2)>3))
##	print f
##	plt.figure()
##	plt.plot(np.sin(f/float(M)*np.arange(M)*2*np.pi))
##	
#	(sinefit,est_phase,est_mean) = ls_sine(angles[:,i]-np.mean(angles[:,i]),f)
#	(sinefit1,est_phase1,est_mean1) = ls_sine(angles[:,i+1]-np.mean(angles[:,i+1]),f)
#	
#	plt.figure()
#	plt.plot(angles[:,i]-np.mean(angles[:,i]),'b--')
#	plt.plot(sinefit,'b')
#	plt.plot(angles[:,i+1]-np.mean(angles[:,i]),'r--')
#	plt.plot(sinefit1,'r')
#	
##	xcorr = np.correlate(angles[:,i]-np.mean(angles[:,i]), angles[:,i+1]-np.mean(angles[:,i]), "full")
##	
##	plt.figure()
##	plt.plot(xcorr)
##	
##	print np.argmax(xcorr)
##	print np.argmax(np.correlate(angles[:,i]-np.mean(angles[:,i]), angles[:,i]-np.mean(angles[:,i]), "full"))
#	
#	print
#	print est_phase
#	print est_phase1
#	print est_phase-est_phase1

#	
#	plt.show()
#	exit(0)
#	
#	
#	p = np.arctan2(y[0,:]-y[5,:], x[0,:]-x[5,:])	#Get orientation of the head of the worm
#	
##	vx=np.mean(np.diff(x[0:5,:],axis=1),axis=0)		#Get x velocity of the head of the worm
##	vy=np.mean(np.diff(y[0:5,:],axis=1),axis=0)		#Get y velocity of the head of the worm
#	vx = np.diff(np.mean(x[0:5,:],axis=0))
#	vy = np.diff(np.mean(y[0:5,:],axis=0))
##	vx=np.mean(np.diff(x[:,:],axis=1),axis=0)		#Get x velocity of the head of the worm
##	vy=np.mean(np.diff(y[:,:],axis=1),axis=0)		#Get y velocity of the head of the worm
#	w = np.arctan2(vy, vx)							#Get orientation of the head's velocity
#	
#	dp=np.diff(np.unwrap(p))
#	dw=w-p[0:-1]	
#	
#	va=(-np.sin(dp)>0).astype(int)*2-1	#+1 turing right, -1 turning left
#	vt=(np.cos(dw)>0)*2-1	#+1 moving forward, -1 moving backwards
#	

##	plt.figure()
##	plt.plot(np.cos(dw))
##	plt.show()
#	print np.mean(vt)
#	print np.mean(np.cos(dw))
#	
#	for i in range(2):
#		t=np.random.randint(N0-1)
#		plt.figure()
#		plt.plot(x[:,t],y[:,t])
#		plt.plot(x[0,t],y[0,t],'o')
#		plt.plot(x[:,t+1],y[:,t+1])
#		plt.plot(x[0,t+1],y[0,t+1],'o')
#		print
#		print va[t]
#		print vt[t]
#	plt.show()
#	exit(0)

#	return vt,va

def compute_eigenworms():
	
	t,x,y=get_positions(0)
	for i in np.arange(1,5):
		t1,x1,y1=get_positions(i)
		x=np.concatenate((x, x1), axis=1)
		y=np.concatenate((y, y1), axis=1)
		t=np.concatenate((t, t1), axis=0)
	N=len(t)
			
	dx = np.diff(x[1:-1], n=1, axis=0)
	dy = np.diff(y[1:-1], n=1, axis=0)
	angles = np.arctan2(dy, dx)
	angles -= np.mean(angles,axis=0)
	
	C = np.cov(angles)
	w,v = np.linalg.eig(C)
	return w,v
	
def get_eigenworms(sheet_index=0):


	M=102
	t=100
	t,x,y=get_positions(sheet_index)

	(t0,x0,y0)=get_center_position(sheet_index)
	N0=len(t0)
	
	x1=np.zeros((M,N0))
	y1=np.zeros((M,N0))
	for i in range(M):
		x1[i,:]=np.interp(t0, t, x[i,:])
		y1[i,:]=np.interp(t0, t, y[i,:])
	
	x=x1+x0*1000
	y=y1+y0*1000

	t=t0
	N=N0
	
	
	w,v=compute_eigenworms()
	
	dx = np.diff(x[1:-1], n=1, axis=0)
	dy = np.diff(y[1:-1], n=1, axis=0)
	angles = np.arctan2(dy, dx)
	angles -= np.mean(angles,axis=0)
	
	a=np.zeros((3,N))
	for i in range(3):
		for t in range(N):
			a[i,t]=np.sum(angles[:,t]*v[:,i])
	return a
	
#	plt.figure()
#	plt.plot(np.cumsum(w)[0:8]/np.sum(w))
#	plt.axis([0,7,0,1])
#	
#	plt.figure()
#	for i in range(4):
#		plt.plot(v[:,i])
		
		
#	t = np.random.randint(N)
#	print angles[i]
#	for i in range(10):
#		print np.sum(angles[:,t]*v[:,i])
#		
#	a=np.zeros((5,N))
#	for i in range(5):
#		for t in range(N):
#			a[i,t]=np.sum(angles[:,t]*v[:,i])
#	w= np.arctan2(a[1,:], a[2,:])
#	plt.figure()
#	plt.plot(a[0,:])
	
#	plt.figure()
#	plt.plot(a[1,:])
#	plt.plot(a[2,:])
#	
#	bins=25
#	amax=2.5
#	W=np.zeros((bins,bins))
#	for t in range(N):
#		i1=np.floor((a[1,t]+amax)*bins/(2*amax))
#		i2=np.floor((a[2,t]+amax)*bins/(2*amax))
#		if i1>=0 and i2>=0 and i1<bins and i2<bins:
#			W[i1,i2]+=1
#	plt.figure()
#	plt.imshow(W,interpolation='nearest')
	

#	plt.figure()
#	plt.plot(w)
#	plt.figure()
#	plt.plot(np.diff(w))

#	w1= np.arctan2((a[1,:]>0)*2-1, (a[2,:]>0)*2-1)
#	
#	w3=(np.abs(a[0,:]**2) > np.mean(np.abs(a[0,:]**2))).astype(int)
#	print 'turning',np.mean(w3)
#	print 'forward',np.mean(np.diff(w[w3[0:-1]==0])>0)*np.mean(1-w3)
#	print 'backwards',np.mean(np.diff(w[w3[0:-1]==0])<0)*np.mean(1-w3)
#	
#	plt.figure()
#	plt.plot(np.sin(np.diff(w1)))
#	
#	print 'forward',np.mean(np.sin(np.diff(w1))>0)
#	print 'backwards',np.mean(np.sin(np.diff(w1))<0)
##	print list((a[1,:]>0).astype(int) + 2*(a[2,:]>0).astype(int))
#	plt.show()
#	
def discretize(x):
	return (x>0).astype(int)*2-1

def plot_movement(sheet_index=0):

	Ml,Mr=get_movement(0)
	
#	plt.figure()
#	plt.plot(Mr)
#	plt.plot(Ml)
#	
#	plt.figure()
#	plt.plot(Mr>0)
#	plt.plot(Ml>0)
	
	print(np.mean((Mr>0)*(Ml>0)))
	print(np.mean((Mr<0)*(Ml<0)))
	print(np.mean((Mr>0)*(Ml<0)))
	print(np.mean((Mr<0)*(Ml>0)))
	
	M=102
	t=100
	t,x,y=get_positions(sheet_index)

	(t0,x0,y0)=get_center_position(sheet_index)
	N0=len(t0)
	
	x1=np.zeros((M,N0))
	y1=np.zeros((M,N0))
	for i in range(M):
		x1[i,:]=np.interp(t0, t, x[i,:])
		y1[i,:]=np.interp(t0, t, y[i,:])
	
	x=x1+x0*1000
	y=y1+y0*1000

	t=t0
	N=N0
	
	t=np.random.randint(N-1)
	plt.figure()
	color='b'
	for i in range(8):
		if i>0:
			color='r'
		plt.plot(x[:,t+i],y[:,t+i],color)
		plt.plot(x[0,t+i],y[0,t+i],color+'o')
		
	plt.figure()
	plt.plot(np.mean(x,axis=0),np.mean(y,axis=0))
	
	plt.show()
	exit(0)
	
	return Ml,Mr

if __name__ == '__main__':

#	(neural_activation,behavior) = get_neural_activation()
#	print neural_activation
#	print behavior

#	vt,va=plot_movement(0)
#	na,b=get_neural_activation(0)
#	
#	vt=discretize(vt)
#	va=discretize(va)
#	print np.mean(vt==1)
#	
#	print np.mean(vt[b[0:-1]==1]==-1)
	a=get_eigenworms()
	w1= np.arctan2((a[1,:]>0)*2-1, (a[2,:]>0)*2-1)
	
	w3=(np.abs(a[0,:]**2) > np.mean(np.abs(a[0,:]**2))).astype(int)
	
	plt.figure()
	plt.plot(np.sin(np.diff(w1)))
	
	print('forward',np.mean(np.sin(np.diff(w1))>0))
	print('backwards',np.mean(np.sin(np.diff(w1))<0))
	
	
