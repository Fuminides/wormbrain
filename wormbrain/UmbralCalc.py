# -*- coding: utf-8 -*-
"""
@author: Javier Fumanal Idocin
"""
import worm
import AnalyzeModel

import numpy as np
import matplotlib.pyplot as plt

from kinetic_ising import bool2int
from itertools import combinations
from sklearn.feature_selection import SelectKBest, f_classif

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
        if anterior != None:
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
            total += np.sum(AnalyzeModel.transmision_entropia(muestra))/muestra.shape[1]
    
    return -(total)

def cap_calorifica(model, tam = 1000):
    total = 0
    model.generate_sample(1, booleans = True)[0]
    for i in range(tam):
        model.GlauberStep()
        h = model.H()
        B = 1/model.T
        total += h**2 * B**2 / np.cosh(h*B)**2 + B*(model.s*h - np.dot(model.s,h))*(B*h*np.tanh(B*h)-np.log(2*np.cosh(B*h)))
        
    total /= tam
    return  np.mean(total)
        
        
def calculate_entropy_ising(ising,tamano_muestra=1000, transiciones = False):
    '''
    Genera una muestra aleatoria de un ising y calcula la entropia de la misma
    '''
    if (transiciones == True):
        return entropia_transiciones(ising, tam=tamano_muestra)
    else:
        muestra = ising.generate_sample(tamano_muestra)
        return entropia(cuenta_transiciones(muestra))
        


def entropia_temperatura(ising, temperaturas=10**np.arange(-1,1.1,0.1), tamano_muestra=1000):
    '''
    Devuelve la entropia del sistema ising para cada temperatura del rango dado.
    (No modifica la temperatura del sistema original)
    '''
    entropias = np.zeros(len(temperaturas))
    temperatura_original = ising.T
    
    for n in np.arange(0,len(temperaturas)):
        ising.T = temperaturas[n]
        entropias[n] = calculate_entropy_ising(ising, transiciones=True)
        
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
    muestra = model.generate_sample(5000, booleans = True)
    (total_m, indv_m) = entropia_completa(original)
    (total_t, indv_t) = entropia_completa(muestra)
    
    return (indv_t - total_t) / (indv_m - total_m)
############################################

def entropiaKneuronas(gusano, k, normalizar=False):
    '''
    Devuelve la media de entropia de cada k combinacion de neuronas del gusano.
    '''
    cuenta_estados = {}
    (neural_activation_original,behaviour)=worm.get_neural_activation(gusano)
    size = neural_activation_original.shape[1] #Numero de dimensiones
    rango = np.arange(0.1,0.3,0.01).tolist()
    registro_entropias = np.zeros(np.size(rango))
    permutaciones = list(combinations(range(size), k))
    neural_activation = AnalyzeModel.umbralizar(neural_activation_original, 0)
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
    for gusano in [0,1,2,3,4]:
        print("Para gusano: %s"%(gusano+1))
        (neural_activation,behaviour)=worm.get_neural_activation(gusano)
        T = neural_activation.shape[0] #Numero de muestras
        registro_entropias = np.zeros(np.arange(0.1,2,0.1).shape[0])
        limit_neuronas = int(np.log2(T))
        cuenta_estados = {}
        neural_activation = SelectKBest(f_classif, k=limit_neuronas).fit_transform(neural_activation, behaviour)
                
        for n in np.arange(0,2,0.1).tolist() + [2,3,4,5]: 
            activaciones = AnalyzeModel.umbralizar(neural_activation, n)
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
    
