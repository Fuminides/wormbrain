# -*- coding: utf-8 -*-
"""
@author: Javier Fumanal Idocin
"""
import worm
import warnings

import analyze_model as am

import numpy as np
import matplotlib.pyplot as plt
import ITtools as it
import ising_recovery as ir


from kinetic_ising import bool2int
from itertools import combinations
from sklearn.feature_selection import SelectKBest, f_classif
from numba import jit
from itertools import permutations

######################################
# FUNCIONES
######################################
def cuenta_estado(activaciones):
    '''
    Cuenta el numero de veces que aparece cada estado en la muestra, y lo devuelve
    en forma de diccionario.
    
    activaciones -- array a analizar. Cada muestra debe ser un entero
    '''
    cuenta_estados = {}
    for estado in activaciones:
        convertido = str(estado)
        if convertido in cuenta_estados:
           cuenta_estados[convertido] = cuenta_estados[convertido] + 1                   
        else:
           cuenta_estados[convertido] = 1
    return cuenta_estados

def cuenta_transiciones(activaciones):
    '''
    Cuenta el numero de veces que aparece cada estado en la muestra, y lo devuelve
    en forma de diccionario.
    
    activaciones -- array a analizar. Cada muestra debe ser un entero
    '''
    cuenta_estados = {}
    anterior = None
    for estado in activaciones:
        if not (anterior is None):
            convertido = str(estado) + str(anterior)
            if convertido in cuenta_estados:
               cuenta_estados[convertido] = cuenta_estados[convertido] + 1                   
            else:
               cuenta_estados[convertido] = 1
               
        anterior = estado
        
    return cuenta_estados

def entropia(estados):
    '''
    Calcula la entropia de una lista de ocurrencias.
    
    estados -- numero de ocurrencias de una serie de eventos.
                NOTA: ocurrencias, no probabilidades.
    '''
    suma = 0
    normalizacion = 0
    for n in estados.keys():
        normalizacion = normalizacion + estados[n]
        
    for n in estados.keys():
        suma = suma + (estados[n]/float(normalizacion)) * np.log2(estados[n]/float(normalizacion))
        
    return -suma

def entropia_transiciones(model, tam = 1000, glauber=False):
    '''
    Calcula la entropia de las transiciones que genera un modelo.
    (Solo tiene sentido con el ising cinetico)
    
    model -- modelo cinetico
    muestra -- muestra a estudiar
    '''
    total = np.zeros(model.J.shape[0])
    s = model.generate_sample(1, booleans = True)[0]
    if not glauber:
        for i in range(tam):
            model.GlauberStep()
            
            h= model.H() * (1/model.T)
            total += sum(h*np.tanh(h) - np.log(2*np.cosh(h)))

        total /= tam
        total = np.mean((total))
    else:
        for i in range(tam):
            muestra = model.generate_sample(3, state=s, booleans = True)
            s = muestra[2]
            total += np.sum(transmision_entropia(muestra))/muestra.shape[1]
    
    return -(total)

def cap_calorifica(model, tam = 5000):
    '''
    Calcula la capacidad calorifica de un sistema.
    
    model -- modelo a analizar.
    tam -- tamanyo de la muestra con la que calcular el modelo.
    '''
    total = 0
    model.generate_sample(1, booleans = True)[0]
    for i in range(tam):
        model.GlauberStep()
        h = model.H()
        B = 1/model.T
        total += h**2 * B**2 / np.cosh(h*B)**2 + B*(model.s*h - np.dot(model.s,h))*(B*h*np.tanh(B*h)-np.log(2*np.cosh(B*h)))
        
    total /= tam
    return  np.mean(total)
        
        
def calculate_entropy_ising(ising,tamano_muestra=None, transiciones = False):
    '''
    Genera una muestra aleatoria de un ising y calcula la entropia de la misma
    '''
    if (transiciones == True):
        if tamano_muestra is None:
            minimum = 1000
            maximum = 50000
            tamano_muestra = 2**ising.size
            if tamano_muestra > maximum:
                tamano_muestra = maximum
            if tamano_muestra < minimum:
                tamano_muestra = minimum

        return entropia_transiciones(ising, tam=tamano_muestra)
    else:
        muestra = ising.generate_sample(tamano_muestra)
        return entropia(cuenta_transiciones(muestra))
        


def entropia_temperatura(ising, temperaturas=10**np.arange(-1,1.1,0.1), tamano_muestra=10000,trans=True):
    '''
    Devuelve la entropia del sistema ising para cada temperatura del rango dado.
    (No modifica la temperatura del sistema original)
    '''
    entropias = np.zeros(len(temperaturas))
    temperatura_original = ising.T
    
    for n in np.arange(0,len(temperaturas)):
        ising.T = temperaturas[n]
        entropias[n] = calculate_entropy_ising(ising, transiciones=trans, tamano_muestra=5000)
        
    ising.T = temperatura_original
    return entropias

def entropia_muestra(conjunto, transiciones, normalizar=False):
    '''
    Devuelve la entropia de un conjunto, calculandola a partir de 
    
    Conjunto -- conjunto del que calcular la entropia
    transiciones -- indica si se quiere medir la entropia de los cambios de estado
    normalizar -- ajusta el valor entre 0 y 1 o no
    '''
    if (not transiciones):
        result = entropia(cuenta_estado(conjunto))
    else:
        result = entropia(cuenta_transiciones(conjunto))
        
    if normalizar:
        result /= max(result)
    
    return result

def entropia_completa(muestra):
    '''
    Calcula la entropia de una muestra, y la suma de las entropias individuales
    de sus variables.
    '''
    e_muestra = entropia(cuenta_estado(muestra))
    e_individuales = 0
    for i in range(muestra.shape[1]):
        e_individuales += entropia(cuenta_estado(muestra[:,i]))
        
    return e_muestra, e_individuales

def correlaciones_capturadas(model, original):
    '''
    Calcula el porcentaje de correlaciones capturadas en el modelo con respecto
    a la muestra de entrenamiento del mismo.
    '''
    muestra = model.generate_sample(booleans = True)
    (total_m, indv_m) = entropia_completa(original)
    (total_t, indv_t) = entropia_completa(muestra)
    
    return (indv_t - total_t) / (indv_m - total_m)

@jit
def transmision_entropia(muestra, tiempo=1):
    '''
    Calcula la transferencia de entropia para todas las combinaciones de 
    dimensiones posibles de la muestra a un tiempo t una de la otra.
    
    muestra -- actividad neuronal discretizada a estudiar.
    tiempo -- distancia temporal a la que se quiere estudiar.
    '''
    dimensiones = muestra.shape[1]
    permutaciones = list(permutations(np.arange(0,dimensiones),2))
    resultados = np.zeros([dimensiones, dimensiones])-1
    
    for permutacion in permutaciones:
        resultados[permutacion[0],permutacion[1]] = it.TransferEntropy(muestra[:,permutacion[0]]+0, muestra[:,permutacion[1]]+0, r = tiempo)
        
    for i in np.arange(0,dimensiones):
        for j in np.arange(0,dimensiones):
            if i==j:
                resultados[i,i] = 0
            
    
    return resultados

def transmisiones_entropia(muestra, rango=np.arange(1,31), verboso = True, guardar = False):
    '''
    Calcula la transferencia de entropia para todas las combinaciones de 
    dimensiones posibles de la muestra en un rango de tiempos.
    Devuelve ademas la suma de esta misma para cada t distinto.
    
    muestras -- actividad neuronal a analizar.
    rango -- tiempos en los que analizar.
    verboso -- muestra por pantalla el T en calculo si True.
    guardar -- si true, guarda el resultado y una imagen del mismo con IsingRecovery.
    '''
    resultados = []
    sumas_entropia = np.zeros(len(rango))
    for i in rango:
        if verboso:
            print("Con T = " + str(i))
            
        resultados.append(transmision_entropia(muestra, i))
        sumas_entropia[i-1] = np.sum(resultados[i-1])
        
    if guardar:
        for i in np.arange(len(rango)):
            ir.save_image(resultados[i], "T" + str(i+1))
            ir.save_results(resultados[i], "T" + str(i+1) + "_datos.dat")
            
    return resultados, sumas_entropia

############################################
# DEPRECATED
############################################
def deprecation(message):
    warnings.warn(message, DeprecationWarning, stacklevel=1)
    
def entropiaKneuronas(gusano, k, normalizar=False):
    '''
    Devuelve la media de entropia de cada k combinacion de neuronas del gusano.
    '''
    deprecation("Usa la formula estandar!")
    
    cuenta_estados = {}
    (neural_activation_original,behaviour)=worm.get_neural_activation(gusano)
    size = neural_activation_original.shape[1] #Numero de dimensiones
    rango = np.arange(0.1,0.3,0.01).tolist()
    registro_entropias = np.zeros(np.size(rango))
    permutaciones = list(combinations(range(size), k))
    neural_activation = am.umbralizar(neural_activation_original, 0)
    print("Numero de permutaciones a calcular: ", len(permutaciones))
    for neuronas in permutaciones:
        indice = 0
        activaciones = neural_activation[:,list(neuronas)]
        for estado in range(np.size(activaciones,0)):
            convertido = bool2int(activaciones[estado,:])
            if convertido in cuenta_estados:
                cuenta_estados[convertido] = cuenta_estados[convertido] + 1                   
            else:
                cuenta_estados[convertido] = 1
       
        registro_entropias[indice] = entropia(cuenta_estados)
        indice = indice + 1
        cuenta_estados.clear()
    
   
    plt.plot(rango,registro_entropias)
    print()
    if normalizar:
        return np.divide(registro_entropias, len(permutaciones))
    else:
        return registro_entropias

def kMejores():
    '''
    Realiza el estudio de los mejores umbrales para cada gusano, usando las
    k neuronas mas correladas con su comportamiento, donde ese k es el mayor 
    k tal que 2^k<= numero muestras
    '''
    deprecation("Usa la formula estandar!")
    for gusano in [0,1,2,3,4]:
        print("Para gusano: %s"%(gusano+1))
        (neural_activation,behaviour)=worm.get_neural_activation(gusano)
        T = neural_activation.shape[0] #Numero de muestras
        registro_entropias = np.zeros(np.arange(0.1,2,0.1).shape[0])
        limit_neuronas = int(np.log2(T))
        cuenta_estados = {}
        neural_activation = SelectKBest(f_classif, k=limit_neuronas).fit_transform(neural_activation, behaviour)
                
        for n in np.arange(0,2,0.1).tolist() + [2,3,4,5]: 
            activaciones = am.umbralizar(neural_activation, n)
            for estado in range(np.size(activaciones,1)):
                convertido = bool2int(activaciones[estado,:])
                if convertido in cuenta_estados:
                    cuenta_estados[convertido] = cuenta_estados[convertido] + 1                   
                else:
                    cuenta_estados[convertido] = 1
            if (n*10 < len(registro_entropias)):
                registro_entropias[int(n*10)] = entropia(cuenta_estados)
            else:
                print("Numero estados: ", len(cuenta_estados))
                print("Con umbral ", n, " :",entropia(cuenta_estados))
                
            cuenta_estados.clear()
        plt.figure() 
        plt.title("Entropia segun umbral. Gusano %s"%(gusano+1))
        plt.plot(np.divide(range(np.size(registro_entropias)),10.0),registro_entropias)
        print()
        

########################################################
    
