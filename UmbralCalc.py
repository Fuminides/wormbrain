# -*- coding: utf-8 -*-
"""
@author: Javier Fumanal Idocin
"""
import worm

import numpy as np
import matplotlib.pyplot as plt

from kinetic_ising import bool2int, bitfield
from itertools import combinations
from sklearn.feature_selection import SelectKBest, f_classif
from AnalyzeModel import umbralizar

######################################
# FUNCIONES
######################################


def cuenta_estado(activaciones):
    '''
    Cuenta el numero de veces que aparece cada estado en la muestra, y lo devuelve
    en forma de diccionario
    '''
    cuenta_estados = {}
    for estado in activaciones:
        convertido = estado
        if convertido in cuenta_estados:
           cuenta_estados[convertido] = cuenta_estados[convertido] + 1                   
        else:
           cuenta_estados[convertido] = 1
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

def calculate_entropy_ising(ising,tamano_muestra=1000):
    '''
    Genera una muestra aleatoria de un ising y calcula la entropia de la misma
    '''
    muestra = ising.generate_sample(tamano_muestra)
    muestra_bool = np.zeros([len(muestra),ising.size])
    for i in np.arange(0,len(muestra)):
        muestra_bool[i] = bitfield(muestra[i], ising.size)
        
    #ocurrencias = cuenta_estado(muestra)
    
    return entropia_muestra(muestra_bool,2)


def entropia_temperatura(ising, temperaturas=np.arange(0,3,0.1), tamano_muestra=1000):
    '''
    Devuelve la entropia del sistema ising para cada temperatura del rango dado.
    (No modifica la temperatura del sistema original)
    '''
    entropias = np.zeros(len(temperaturas))
    temperatura_original = ising.Beta
    
    for n in np.arange(0,len(temperaturas)):
        ising.Beta = temperaturas[n]
        entropias[n] = calculate_entropy_ising(ising)
        
    ising.Beta = temperatura_original
    return entropias


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
            activaciones = umbralizar(neural_activation, n)
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
        
def entropia1neurona(gusano):
    '''
    Devuelve la media de entropia de cada neurona para cada umbral por separado.
    '''
    cuenta_estados = {}
    (neural_activation_original,behaviour)=worm.get_neural_activation(gusano)
    size = neural_activation_original.shape[1] #Numero de dimensiones
    registro_entropias = np.zeros(np.arange(0,2,0.1).shape[0]+4)
    
    for neurona in range(size):
        neural_activation = neural_activation_original[:,neurona]
    
        for n in np.arange(0,2,0.1).tolist() + [2,3,4,5]: 
            activaciones = umbralizar(neural_activation, n)
            for estado in range(np.size(activaciones)):
                convertido = (activaciones[estado])+0
                if convertido in cuenta_estados:
                    cuenta_estados[convertido] = cuenta_estados[convertido] + 1                   
                else:
                    cuenta_estados[convertido] = 1
            if (n*10 < len(registro_entropias)):
                registro_entropias[int(n*10)] += entropia(cuenta_estados)
            else:
                registro_entropias[int(((n-2)*0.1+2)*10)] += entropia(cuenta_estados)
                
            cuenta_estados.clear()
            
    return np.divide(registro_entropias, size)

def entropiaKneuronas(gusano, k, normalizar=True):
    '''
    Devuelve la media de entropia de cada k combinacion de neuronas del gusano.
    '''
    cuenta_estados = {}
    (neural_activation_original,behaviour)=worm.get_neural_activation(gusano)
    size = neural_activation_original.shape[1] #Numero de dimensiones
    rango = np.arange(0.1,0.3,0.01).tolist()
    registro_entropias = np.zeros(np.size(rango))
    permutaciones = list(combinations(range(size), k))
    
    print("Numero de permutaciones a calcular: ", len(permutaciones))
    for neuronas in permutaciones:
        indice = 0
        neural_activation = neural_activation_original[:,list(neuronas)]
        for n in rango:
            activaciones = umbralizar(neural_activation, n)
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

def entropia_conjunto(conjunto, k, umbral, normalizar=False):
    '''
    Devuelve la entropia de un conjunto, calculandola a partir de 
    la suma de entropia de las k combinaciones posibles de su dimensionalidad.
    Conjunto de entrada continuo.
    (Evita resultados estadisticamente no utiles)
    
    Conjunto -- conjunto del que calcular la entropia
    k -- tamano de las combinaciones
    umbral -- tipo/umbral a utilizar segun umbralizar()
    normalizar -- ajusta el valor a la cantidad de combinaciones estudiadas o no
    '''
    cuenta_estados = {}
    size = conjunto.shape[1] #Numero de dimensiones
    registro_entropias = 0
    permutaciones = list(combinations(range(size), k))
    
    print("Numero de permutaciones a calcular: ", len(permutaciones))
    for neuronas in permutaciones:
        neural_activation = conjunto[:,list(neuronas)]
        activaciones = umbralizar(neural_activation, umbral)
        for estado in range(np.size(activaciones,0)):
            convertido = bool2int(activaciones[estado,:])
            if convertido in cuenta_estados:
                cuenta_estados[convertido] = cuenta_estados[convertido] + 1                   
            else:
                cuenta_estados[convertido] = 1
       
        registro_entropias += entropia(cuenta_estados)
        cuenta_estados.clear()
    
  
    if normalizar:
        return registro_entropias/ len(permutaciones)
    else:
        return registro_entropias

def entropia_muestra(conjunto, k, normalizar=True):
    '''
    Devuelve la entropia de un conjunto, calculandola a partir de 
    la suma de entropia de las k combinaciones posibles de su dimensionalidad.
    La entrada debe estar ya discretizada.
    (Evita resultados estadisticamente no utiles)
    
     Conjunto -- conjunto del que calcular la entropia
    k -- tamano de las combinaciones
    normalizar -- ajusta el valor a la cantidad de combinaciones estudiadas o no
    '''
    cuenta_estados = {}
    size = conjunto.shape[1] #Numero de dimensiones
    permutaciones = list(combinations(range(size), k))
    resultado = 0
    num_estados = 0
    for neuronas in permutaciones:
        neural_activation = conjunto[:,list(neuronas)]
        for estado in range(np.size(neural_activation,0)):
            num_estados += 1
            convertido = bool2int(neural_activation[estado,:])
            if convertido in cuenta_estados:
                cuenta_estados[convertido] = cuenta_estados[convertido] + 1                   
            else:
                cuenta_estados[convertido] = 1
       
        resultado += entropia(cuenta_estados)
        cuenta_estados.clear()
    
  
    if normalizar:
        return resultado/ num_estados
    else:
        return resultado

############################################

if __name__ == '__main__':
    pr = entropiaKneuronas(0, 2)
    
