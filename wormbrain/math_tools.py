# -*- coding: utf-8 -*-
"""
@author: Javier Fumanal Idocin
"""
import entropy_metrics

import numpy as np
import matplotlib.pyplot as plt

from kinetic_ising import bool2int, bitfield
from scipy.optimize import curve_fit, fmin
from scipy import stats
from functools import partial
from random import random

##################################################
# Funciones
##################################################
def calcMeanCov(muestra, booleans = True, tiempo_=5, size = 0):
    '''
    Calcula la media y las correlaciones de una muestra
    '''
    if booleans:
        T = muestra.shape[0]
        size = muestra.shape[1]
        ##Calculamos la media y la covarianza de cada neurona
        sample = np.zeros(T)
        for i in range(T):
            sample[i] = (bool2int(muestra[i,:]))
    else:
        T = len(muestra)
        sample = muestra
    
    m1=np.zeros(size)
    D1=np.zeros((size,size))
    s=bitfield(sample[0],size)*2-1
    m1+=s/float(T)
    
    for l in np.arange(tiempo_,T):
        n = sample[l]
        #for t = sample de t + 5 
        sprev=bitfield(sample[l-tiempo_],size)*2-1
        s=bitfield(n,size)*2-1
        m1+=s/float(T)
        for i in range(size):
            D1[:,i]+=s[i]*sprev/float(T-1)
            
    for i in range(size):
        for j in range(size):
                D1[i,j]-=m1[i]*m1[j]
                
    return m1, D1

def color_bar(data):
    '''
    Muestra por pantalla el color bar de un array de numpy.
    '''
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ticks_at = [-abs(data).max(), 0, abs(data).max()]
    cax = ax.imshow(data, interpolation='nearest', 
                    origin='lower', extent=[0.0, 0.1, 0.0, 0.1],
                    vmin=ticks_at[0], vmax=ticks_at[-1])
    cbar = fig.colorbar(cax,ticks=ticks_at,format='%1.2g')
    return cbar

def zipf(n , a):
    '''
    Devuelve un termino de la distribucion de zipf.
    Usar solo para curve_fit.
    '''
    return 1/np.power(n,a)
    
def zipf_sample(n , a, size = None):
    '''
    Devuelve un termino de la distribucion de zipf.
    Sample ready. Usar siempre menos para curve_fit.
    '''
    if size is None:
        return zipf(n,a)
    else:
        resultado = np.zeros(size)
        for i in range(size):
            resultado[i] = 1/int(random()*100)**a
            
        return resultado

def zipf_approximation(real_data, transiciones=False, verboso = False):
    '''
    Genera una distribucion de Zipf lo mas parecida posible a la muestra original,
    y le aplica el test de Kolmogorov.
    
    real_data -- muestra a analizar
    transiciones -- si se desea analizar las transciones entre estados en vez de los estados en si
    verboso -- si se quiere mostrar la grafica por pantalla o no.
    '''
    if not transiciones:
        norm_prob = distribucion_probabilidad_estados(real_data)
    else:
        norm_prob = distribucion_probabilidad_transiciones(real_data)
    
    norm_prob /= np.max(norm_prob)
    funcion, extra = curve_fit(zipf,np.arange(1,len(norm_prob)+1),norm_prob)
    a_ = funcion[0]
    
    if a_ <= 0.0:
        print("ERROR: algo mal ha ocurrido al aproximar la funcion. a<=0.0")
        
    zipf_fitted = partial(zipf_sample, a=a_)
    
    test1 = stats.ks_2samp(norm_prob, zipf_fitted(np.arange(1,real_data.shape[0])))
    test2 = stats.kstest(norm_prob, zipf_fitted)
    
    if verboso:
        plt.figure()
        plt.semilogx(np.log(norm_prob))
        plt.semilogx(np.log(zipf_sample(np.arange(1,len(real_data)+1),a=a_)),'r-')
        
    return test1, test2, a_

def sigmoidal(x,x0,k):
    '''
    Calcula la funcion sigmoidal de un numero
    '''
    return 1 / (1+np.exp(-k*(x-x0)))

def inversa_sigmoidal(y,x0,k):
    '''
    Devuelve la inversa de la sigmoidal
    '''
    return np.log(1/y - 1)/(-k) + x0

def derivada_sigmoidal(x,x0,k):
    '''
    Calcula la derivada de la funcion sigmoidal de un numero
    '''
    return np.exp(-k*(x-x0)) /( (1+np.exp(-k*(x-x0)))**2)

def aproximacion_sigmoidal(x_func, entropias_calc, verboso=True, montecarlo=6):
    '''
    Devuelve, dada una serie de puntos [x,y] una funcion que los aproxime
    junto con su derivada, utilizando montecarlo.
    
    x_func -- x de los puntos
    entropias_calc -- y de los puntos (se normaliza para el calculo)
    grado -- grado de la funcion a aproximar
    montecarlo -- numero de puntos a utilizar para la aproximacion
    verboso -- si True, saca por pantalla una figura con el resultado.
                Nota: La funcion derivada aparecera normalizada en la figura
    '''
    eleccion = np.random.choice(int(len(entropias_calc)), montecarlo, replace=False)
    eleccion = np.sort(eleccion)
    x_approx = x_func[eleccion]
    escala = np.max(entropias_calc)
    entropias_calc = entropias_calc/escala
    montecarlo_sample = entropias_calc[eleccion]
    popt, pcov = curve_fit(sigmoidal, x_approx, montecarlo_sample)
    
    y_new = np.zeros(montecarlo)
    y_derivada  = np.zeros(montecarlo)
    
    indice = 0
    for x in x_approx:
        y_new[indice] = sigmoidal(x,*popt)
        y_derivada[indice] = derivada_sigmoidal(x,*popt)
        indice += 1
    
    maximo = fmin(lambda x: -derivada_sigmoidal(x, *popt), 0)
    
    if verboso:
        plt.figure()
        plt.plot(x_func, entropias_calc*escala)
        plt.plot(x_approx, montecarlo_sample*escala,'ro')
        plt.plot(x_func, y_new*escala)
        plt.plot(x_func, y_derivada*escala)
        plt.plot(maximo, sigmoidal(maximo,*popt)*escala, 'yo')
    
    return popt, maximo, eleccion, escala

def derivada_maxima_aproximada(indices_x,indices_y):
    '''
    Devuelve el punto de maximo crecimiento de una funcion de forma aproximada.
    
    El punto se calcula calculando la pendiente entre cada par de puntos
    consecutivos, y eligiendo el punto medio entre aquellos que posean la
    mayor.
    '''
    maximo = 0
    
    for i in np.arange(1,len(indices_y)):
        avanzado = abs(indices_y[i]-indices_y[i-1])
        espacio = indices_x[i]-indices_x[i-1]

        if avanzado/espacio > maximo:
            resultado = (indices_x[i-1] + indices_x[i])/2 
            maximo = avanzado/espacio
            valor = indices_y[i-1] + (resultado-indices_x[i-1])*maximo
    
    return resultado,valor

def distribucion_probabilidad_estados(muestras, verboso = False):
    '''
    Devuelve ordenadas las probabilidades de cada estado de la muestra de mayor a menor.
    
    muestras -- array de muestras
    entero -- indica si las muestras van en forma de enteros o de array de bools
    verboso -- ensenya una grafica con las probabilidades (x logaritmo)
    '''
    ocurrencias = list(entropy_metrics.cuenta_estado(muestras).values())
        
    ocurrencias.sort(reverse=True)
    ocurrencias = np.divide(ocurrencias,sum(ocurrencias))
    
    if verboso:
        plt.figure()
        plt.plot(np.log(np.arange(0,len(ocurrencias))),ocurrencias)
        
    return ocurrencias

def distribucion_probabilidad_transiciones(muestras, verboso = False):
    '''
    Devuelve ordenadas las probabilidades de cada estado de la muestra de mayor a menor.
    
    muestras -- array de muestras
    entero -- indica si las muestras van en forma de enteros o de array de bools
    verboso -- ensenya una grafica con las probabilidades (x logaritmico)
    '''
    ocurrencias = list(entropy_metrics.cuenta_transiciones(muestras).values())
    ocurrencias.sort(reverse=True)
    ocurrencias = np.divide(ocurrencias,sum(ocurrencias))
    
    if verboso:
        plt.figure()
        plt.plot(np.log(np.arange(0,len(ocurrencias))),ocurrencias)
        
    return ocurrencias