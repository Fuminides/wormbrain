# -*- coding: utf-8 -*-
"""
@author: Javier Fumanal Idocin
"""
#!/usr/bin/env python
import sys
import smtplib
import worm
import time
import scipy
import UmbralCalc

import ising as isng
import matplotlib.pyplot as plt
import numpy as np
import ITtools as IT
import IsingRecovery as IR
import networkx as nx
import plotly.plotly as py


from kinetic_ising import ising, bitfield, bool2int
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from numba import jit
from scipy.optimize import curve_fit
from RedClasificador import entrenar_clasificador, process_labels
from IsingRecovery import save_isings, restore_ising
from itertools import permutations
from random import random
from plotly.graph_objs import *



sys.path.insert(0, '..')

#######################################################
#Parametros del programa
#######################################################
gusano=0
error=1E-3
variabilidad = 0.85
barajeo = None

##################################################
# Funciones
##################################################

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
    
    maximo = scipy.optimize.fmin(lambda x: -derivada_sigmoidal(x, *popt), 0)
    
    if verboso:
        plt.figure()
        plt.plot(x_func, entropias_calc*escala)
        plt.plot(x_approx, montecarlo_sample*escala,'ro')
        plt.plot(x_func, y_new*escala)
        plt.plot(x_func, y_derivada/derivada_sigmoidal(maximo,*popt)*escala)
        plt.plot(maximo, sigmoidal(maximo,*popt)*escala, 'yo')
    
    return popt, maximo, eleccion, escala

def puntuar(resultado, maximo, parametros):
    '''
    Puntua del 0 al 10 como de cerca esta un valor de una sigmoidal de su
    punto de criticalidad.
    
    resultado -- valor a puntuar
    maximo -- valor donde la derivada de la sigmoidal es maxima
    parametros -- valores de ajuste de la sigmoidal (x0 y k)
    '''
    aux = derivada_sigmoidal(inversa_sigmoidal(resultado,*parametros),*parametros) / derivada_sigmoidal(maximo,*parametros) * 10
    if aux > 10:
        aux = 10 - aux%10
    
    return aux

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


def compresion(neural_activation, behavior, comprimir):
    '''
    Aplica distintas tecnicas para reducir el numero de neuronas.
    
    neural_activation -- el conjunto a reducir
    behaviour -- el comportamiento esperado del gusano (label)
    comprimir --
         0->Sin compresion
         1->Usando PCA 
         2->Coge las mas correladas con las labels
         >=3 -> Coge ese mismo numero de neuronas aleatorias
    '''
    global barajeo
    if comprimir == 1:
        pca = PCA()
        pca.fit(neural_activation)
        variabilidades = pca.explained_variance_ratio_
        acum = 0
        x_axis = []
        i = -1
        busca = True
        for v in variabilidades:
            i += 1
            acum += v
            if len(x_axis) > 0:
                x_axis = x_axis + [x_axis[-1] + v]
            else:
                x_axis = [v]
            if busca and (acum >= variabilidad):
                componentes = i
                busca = False
                
        #Ensenamos figura con los valores propios (Por si queremos usar regla del codo)
        plt.plot(x_axis, variabilidades, 'r-')
        plt.axis([x_axis[0],1.0,0.0,variabilidades[0]])
        plt.ylabel('Valores propios')
        plt.show()
        
        pca = PCA(n_components=componentes)
        neural_activation = pca.fit_transform(neural_activation)

    elif comprimir == 2:
        ##Tree-base feature selector
        clf = ExtraTreesClassifier()
        clf.fit(neural_activation, behavior)
        model = SelectFromModel(clf, prefit=True)
        neural_activation = model.transform(neural_activation)
        
    elif comprimir >= 3:
        ##Eleccion aleatoria de neuronas
        if barajeo==None:
            barajeo = np.arange(0,neural_activation.shape[1])
            np.random.shuffle(barajeo)
            
        neural_activation = neural_activation[:,barajeo[0:comprimir]]

    return neural_activation


def umbralizar(neural_activation, umbral, verboso=False):
    '''
    Umbraliza las neuronas mediante diversas tecnicas.
    
    neural_activation -- set de neuronas
    umbral --
        <2 -> Se usa el propio valor del argumento como umbral (valor fijo)
        =2 -> Media de todas las neuronas
        =3 -> Media de cada neurona es su propio umbral
        =4 -> Mediana de todas las neuronas
        =5 -> Mediana de cada neurona es su propio umbral
        =6 -> Umbral igual a 0
    verboso -- si cierto, muestra un histograma con el numero de activaciones 
                de cada neurona
    filtrado -- si True, se pasa un filtro de paso alto
    '''
   
    if np.size(neural_activation.shape) > 1:
        size = neural_activation.shape[1]
        T = neural_activation.shape[0]
    else:
        size = 0
        T = len(neural_activation)
        
    if umbral==2:
        umbral=np.mean(neural_activation) #Cogemos las media de todas las neuronas como umbral
    elif umbral == 3:
        if size == 0:
            media_columnas = np.mean(neural_activation)
            umbral = np.zeros(T)
        else:
            media_columnas = np.mean(neural_activation, axis = 0)
            umbral = np.zeros((T,size))
            
        for n in range(T):
            umbral[n] = media_columnas
                  
    elif umbral == 4:
        umbral = np.median(neural_activation)
        
    elif umbral == 5:
        if size == 0:
            mediana_columnas = np.median(neural_activation)
            umbral = np.zeros(T)
        else:
            mediana_columnas = np.median(neural_activation, axis = 0)
            umbral = np.zeros((T,size))
        for n in range(T):
            umbral[n] = mediana_columnas
    elif umbral == 6:
        umbral = 0
    if verboso:
        #Visualizar grafico de barras             
        act_hist = np.sum(np.greater_equal(neural_activation, umbral), axis = 0)
        axes = plt.gca()
        axes.set_ylim([0,T])
        plt.title("Activaciones de cada neurona")
        plt.bar(range(np.size(act_hist)),act_hist)
        
    return np.greater_equal(neural_activation, umbral)


def mandar_aviso_correo(gusano, destino = "javierfumanalidocin@gmail.com"):
    '''
    Manda un correo para avisar del fin del entrenamiento de un gusano.
    Por defecto lo manda a mi cuenta de correo.
    Incluye una serie de practicas muy malas, soy consciente, pero por ahora
    me simplifican la vida.
    '''
    print("Mandando correo...")
    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.starttls()
    server.login("wormbraindummy@gmail.com", "wormbraindummy1") #Es una practica horrorosa, lo se
    msg = MIMEMultipart()
    msg['From'] = "wormbraindummy@gmail.com"
    msg['To'] = destino
    msg['Subject'] = "Entrenamiento finalizado"
    body = "Se ha terminado de entrenar el gusano " + gusano
    msg.attach(MIMEText(body, 'plain'))
    
    server.sendmail("wormbraindummy@gmail.com", destino, msg.as_string())
    server.quit()
    
def crear_clasificador(gusano, filename = 'defecto', umbral = 4):
    '''
    Crea y guarda en un fichero un clasificador que detecta el comportamiento
    del gusano a partir de su actividad neuronal.
    
    gusano -- numero del gusano a entrenar.
    filename -- nombre del fichero donde guardar el clasficador.
    umbral -- umbral a utilizar para discretizar la actividad neuronal.
    '''
    (neural_activation,behavior)=worm.get_neural_activation(gusano)
    behavior = np.maximum(behavior, np.zeros(behavior.size))
    
    porcentaje_train = 0.8
    
    barajeo = np.arange(behavior.size)
    np.random.shuffle(barajeo)
    muestras = umbralizar(neural_activation[barajeo],5)
    muestras_l = behavior[barajeo]
    corte = int(porcentaje_train*muestras.shape[0])
    entrenar_clasificador(muestras[0:corte,:], muestras[corte+1:,:],process_labels(muestras_l[0:corte]),  process_labels(muestras_l[corte+1:]), filename =str(gusano)+'_gusano.json')

    
@jit
def train_ising(kinectic=True, comprimir = 0, umbral = 0.17, aviso_email = True, gusanos = np.arange(0,5), filename = 'filename_ising.obj', temperatura = 1):
    '''
    Entrena un modelo de ising para cada uno de los gusanos dados.
    Los escribe en un fichero, ademas de devolverlos como resultado.
    Usa numba para optimizar el codigo.
    
    kinetic -- Si True, usara el modelo de Ising cinetico. (Suele ser mejor)
    comprimir -- indica el tipo de compresion a utilizar para la 
                    dimensionalidad de las neuronas. (Consultar: compresion())
    umbral -- umbral a utilizar. Parametro funciona igual que para: umbralizar()
    aviso_email -- si True, avisara por correo electronico cuando cada gusano
                    termine de entrenar
    gusano -- array con los indices de los gusanos a entrenar
    temperatura -- temperatura a la que poner a funcionar el sistema
    '''
    isings = [ising(1)]
    fits = [0.0]
    
    for gusano in gusanos:
        ##Cogemos los datos del gusano
        (neural_activation,behavior)=worm.get_neural_activation(gusano, True)
        neural_activation = compresion(neural_activation, behavior, comprimir)
        
        ##Calculamos la dimension del array de las neuronas.
        size = neural_activation.shape[1] #Numero de dimensiones
        T = neural_activation.shape[0] #Numero de muestras
        
        activaciones = umbralizar(neural_activation, umbral)
        
        ##Calculamos la media y la covarianza de cada neurona
        sample = np.zeros(T)
        for i in range(T):
           sample[i] = (bool2int(activaciones[i,:]))
        
        m1=np.zeros(size)
        D1=np.zeros((size,size))
        s=bitfield(sample[0],size)*2-1
        m1+=s/float(T)
        for n in sample[1:]:
            #for t = sample de t + 5 
            sprev=s
            s=bitfield(n,size)*2-1
            m1+=s/float(T)
            for i in range(size):
                D1[:,i]+=s[i]*sprev/float(T-1)
                
        for i in range(size):
            for j in range(size):
                    D1[i,j]-=m1[i]*m1[j]
                        
        if (kinectic):
            y=ising(size)
            y.T = temperatura
            y.independent_model(m1)
            fit=y.inverse(m1,D1,error,sample)
        else:
           y=isng.ising(size)
           y.T = temperatura
           fit=y.inverse_exact(m1,D1,error)
            
        isings.append(y)
        fits.append(fit)
        
        if aviso_email:
            mandar_aviso_correo(str(gusano+1))
        
        print("Terminado un entrenamiento: " + time.ctime())
    
    print("Escribiendo en fichero... ")
    save_isings(isings[1:], fits[1:], filename)
    
    print("Entrenamientos finalizados. Todo correcto")
    return isings[1:], fits[1:]
    
    
def punto_criticalidad(ising, tipo_compresion, gusano, montecarlo=15):
    '''
    Muestra por pantalla el punto de criticalidad del sistema ising dado.
    La funcion sigmoidal se calcula por defecto y como maximo, con 15 temperaturas aleatorias.
    
    ising -- sistema ising entrenado
    tipo_compresion -- sistema de reduccion de dimensionalidad a utilizar para las neuronas.
                (Usar el mismo que el usado para entrenar el ising)
    gusano -- gusano con el que comparar la entropia. (Usar el mismo que el entrenado con ising)
    montecarlo -- numero de muestras para la aproximacion sigmoidal.
    '''
    entropias_calc = UmbralCalc.entropia_temperatura(ising, tipo_compresion)
    (neural_activation,behavior)=worm.get_neural_activation(0)
    neural_activation = compresion(neural_activation, behavior, tipo_compresion)
    
    plt.figure()
    plt.plot(np.arange(0,1.5,0.1), entropias_calc[0:15])
    plt.plot(mejor_punto, valor,'ro')
    
    resultado = UmbralCalc.entropia_muestra(umbralizar(neural_activation,4), 2)
    plt.axhline(y=resultado, color='r', linestyle='-')
    
    funcion, maximo, muestras = aproximacion_sigmoidal(np.arange(0,1.5,0.1), entropias_calc[0:15], montecarlo=15)
    
def distribucion_probabilidad_estados(muestras, verboso = False):
    '''
    Devuelve ordenadas las probabilidades de cada estado de la muestra de mayor a menor.
    
    muestras -- array de muestras
    entero -- indica si las muestras van en forma de enteros o de array de bools
    verboso -- ensenya una grafica con las probabilidades (x logaritmo)
    '''
    ocurrencias = list(UmbralCalc.cuenta_estado(muestras).values())
        
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
    ocurrencias = list(UmbralCalc.cuenta_transiciones(muestras).values())
    ocurrencias.sort(reverse=True)
    ocurrencias = np.divide(ocurrencias,sum(ocurrencias))
    
    if verboso:
        plt.figure()
        plt.plot(np.log(np.arange(0,len(ocurrencias))),ocurrencias)
        
    return ocurrencias

def buscar_estable(ising, iteraciones = 5000, max_intentos=np.inf):
    '''
    Termina cuando el sistema ising ha llegado a un punto estable.
    Devuelve de forma aproximada el numero de iteraciones que le ha costado
    llegar.
    
    ising -- sistema ising a medir
    iteraciones -- x numero de muestras en cada remesa de muestras
    max_intentos -- numero de remesas maximo a generar
    '''
    estable = True
    intentos = 0
    samples = ising.generate_sample(iteraciones, None);
    valor_final = samples[0]
    
    for i in np.arange(0,iteraciones):
        if (valor_final != samples[i]):
            estable = False
            
    while(not estable and (max_intentos >= intentos)):
        estable = True
        intentos += 1
        samples = ising.generate_sample(iteraciones, ising.s);
        valor_final = samples[0]
    
        for i in np.arange(0,iteraciones):
            if (valor_final != samples[i]):
                estable = False
    
    return intentos*iteraciones, estable

@jit
def calculo_magnetismo(ising, precision = 50, verboso = True):
    '''
    Calcula el numero de intentos necesarios para llevar a un sistema ising a la
    estabilidad para diferentes temperaturas. Tambien devuelve la facilidad relativa para hacerlo 
    en funcion de esta ultima.
    
    ising -- sistema a estudiar
    precision -- la resolucion con la que calcular el numero de intentos a utilizar.
    Ademas, es el numero de muestras que se estudian para medir la estabilidad del sistema.
    Un valor menor de 50 esta muy desaconsejado, ya que lleva a falsos positivos.
    Recomendado: >50 y < 100
    verboso -- muestra una grafica con los resultados
    '''
    rango_estudio = np.arange(0,1.5,0.1)
    T_original =  ising.T
    intentos = np.zeros(15)
    
    for i in np.arange(0,rango_estudio.size):
        print("Con T igual a: " + str(rango_estudio[i]))
        ising.T = rango_estudio[i]
        pruebas = np.zeros(10)
        for z in np.arange(pruebas.size):
            num, exito = buscar_estable(ising, precision, 500)
            if exito:
                pruebas[z] = num
                
        intentos[i] = np.median(pruebas)
        print(intentos[i])
        if intentos[i] == 0:
            break
    
    for i in intentos.size:
        if intentos[i] ==0:
            intentos[i] = np.max(intentos)
       
    if verboso:
        plt.figure()
        facilidades =  (np.max(intentos) - intentos) / np.max(intentos)
        plt.plot(rango_estudio, facilidades)
        plt.ylabel('Facilidad de llevar a punto de estabilidad')
        plt.xlabel('Temperatura')
        
        plt.figure()
        plt.plot(rango_estudio, intentos)
        plt.ylabel('Numero de intentos para estabilizar')
        plt.xlabel('Temperatura')
        
    ising.T = T_original
    return intentos, facilidades

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
        resultados[permutacion[0],permutacion[1]] = IT.TransferEntropy(muestra[:,permutacion[0]]+0, muestra[:,permutacion[1]]+0, r = tiempo)
        
    for i in np.arange(0,dimensiones):
        for j in np.arange(0,dimensiones):
            if i==j:
                resultados[i,i] = 0
            
    
    return resultados

def transmisiones_entropia(muestra, rango=np.arange(1,31), verboso = True, guardar = True):
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
            IR.save_image(resultados[i], "T" + str(i+1))
            IR.save_results(resultados[i], "T" + str(i+1) + "_datos.dat")
            
    return resultados, sumas_entropia

def largest_indices(ary, n):
    """Devuelve los n indices mas grandes de un array de numpy."""
    flat = ary.flatten()
    indices = np.argpartition(flat, -n)[-n:]
    indices = indices[np.argsort(-flat[indices])]
    return np.unravel_index(indices, ary.shape)

def busca_conexiones(muestras, ratio_real = True):
    '''
    Devuelve una matriz con el numero de veces que un termino i,j
    ha pasado el corte para considerado como una conexion.
    
    muestras -- actividad neuronal a analizar en diferentes tiempos.
    '''
    conexiones = []
    ratio_nodo_arco = 1.5703971119133575
    nodos_a_coger = int(muestras[0].shape[0] * ratio_nodo_arco)
    
    for matrix in muestras:
        for i in np.arange(0,matrix.shape[0]):
            for j in np.arange(0,matrix.shape[1]):
                if i==j:
                    matrix[i,j] = 0
        if not ratio_real:            
            umbral = min(np.mean(matrix), np.median(matrix))
            detectadas = np.less_equal(matrix, umbral)
        else:
            indexes = largest_indices(matrix, nodos_a_coger)
            detectadas = np.zeros(matrix.shape)
            detectadas[indexes] = 1
            
        conexiones.append(detectadas)
        
    resultado = np.zeros(muestras[0].shape)
    
    for matrix in conexiones:
        for i in np.arange(0,resultado.shape[0]):
            for j in np.arange(0,resultado.shape[1]):
                if (i!=j) and (matrix[i,j]):
                    resultado[i,j] += 1
                
    
    return resultado

def reconstruir_red(muestra, ratio_real = True, fiabilidad = 1, cut_ciclos = True, verboso=True):
    '''
    A partir de una serie de muestras de transmision de entropia entre neuronas
    reconstruye una aproximacion de las conexiones reales entre neuronas.
    Devuelve un grafo dirigido y la matriz de conexiones.
    
    muestra -- entropia de cada conexion por pares en distintos tiempos.
    fiabilidad -- nos quedaremos 
    verboso -- muestra por pantalla las conexiones obtenidas
    '''
    fiabilidades = busca_conexiones(muestra, ratio_real)
    
    if not ratio_real:
        fiabilidades = fiabilidades*1.0 / len(muestra)
        conexiones = np.greater_equal(fiabilidades, fiabilidad)
    else:
        ratio_nodo_arco = 1.5703971119133575
        nodos_a_coger = int(muestra[0].shape[0] * ratio_nodo_arco)
        indexes = largest_indices(fiabilidades, nodos_a_coger)
        conexiones = np.zeros(fiabilidades.shape)
        conexiones[indexes] = 1
        
    if cut_ciclos:
        conexiones = corta_ciclos(conexiones)
    
    if verboso:
        plt.imshow(conexiones)
        
    g=nx.DiGraph()
    for i in range(len(muestra[0])):
        g.add_node(i, pos = [random(),random()])
     
    for i in range(conexiones.shape[0]):
        for j in range(conexiones.shape[1]):            
            if (i != j) and (conexiones[i,j]):
                print("Añadida conexion: "+ str(i) + ", " + str(j))
                g.add_edge(i,j)
                
    return g, conexiones
        
def dibujar_grafo(G, name):
    '''
    Dibujar el grafo dado usando plotly.
    
    G -- grafo a dibujar
    name -- nombre del grafo/fichero de plotly donde guardarlo.
    
    IMPORTANTE: requiere de un usuario/key ya definidos.
    '''
    pos=nx.get_node_attributes(G,'pos')
    
    dmin=1
    ncenter=0
    for n in pos:
        x,y=pos[n]
        d=(x-0.5)**2+(y-0.5)**2
        if d<dmin:
            ncenter=n
            dmin=d
    
    p=nx.single_source_shortest_path_length(G,ncenter)
    edge_trace = Scatter(
    x=[],
    y=[],
    line=Line(width=0.5,color='#888'),
    hoverinfo='none',
    mode='lines')

    for edge in G.edges():
        x0, y0 = G.node[edge[0]]['pos']
        x1, y1 = G.node[edge[1]]['pos']
        edge_trace['x'] += [x0, x1, None]
        edge_trace['y'] += [y0, y1, None]
    
    node_trace = Scatter(
        x=[],
        y=[],
        text=[],
        mode='markers',
        hoverinfo='text',
        marker=Marker(
            showscale=True,
            # colorscale options
            # 'Greys' | 'Greens' | 'Bluered' | 'Hot' | 'Picnic' | 'Portland' |
            # Jet' | 'RdBu' | 'Blackbody' | 'Earth' | 'Electric' | 'YIOrRd' | 'YIGnBu'
            colorscale='YIGnBu',
            reversescale=True,
            color=[],
            size=10,
            colorbar=dict(
                thickness=15,
                title='Número de conexiones',
                xanchor='left',
                titleside='right'
            ),
            line=dict(width=2)))
    
    for node in G.nodes():
        x, y = G.node[node]['pos']
        node_trace['x'].append(x)
        node_trace['y'].append(y)
    
    for node, adjacencies in enumerate(G.adjacency_list()):
        node_trace['marker']['color'].append(len(adjacencies))
        node_info = '# of connections: '+str(len(adjacencies))
        node_trace['text'].append(node_info)
        
    fig = Figure(data=Data([edge_trace, node_trace]),
             layout=Layout(
                title='<br>' + name,
                titlefont=dict(size=16),
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20,l=5,r=5,t=40),
                annotations=[ dict(
                    text="Fuente : <a href='https://github.com/Fuminides/wormbrain'> https://github.com/Fuminides/wormbrain</a>",
                    showarrow=False,
                    xref="paper", yref="paper",
                    x=0.005, y=-0.002 ) ],
                xaxis=XAxis(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=YAxis(showgrid=False, zeroline=False, showticklabels=False)))

    py.iplot(fig, filename=name)
    
def corta_ciclos_aux(conexiones, visitados, inicio, final):
    '''
    Funcion auxiliar de corta_ciclos(). Vamos, que no la uses.
    '''
    inicial = visitados
    
    for i in range(inicio, final):
        for j in range(conexiones.shape[1]):
    
            if (conexiones[i,j]):
                if j in visitados:
                    conexiones[i,j] = False
                else:
                    visitados += [j]
                    conexiones = corta_ciclos_aux(conexiones, visitados, j, j+1)
                    
        visitados = inicial  
          
    return conexiones

def corta_ciclos(conexiones):
    '''
    Corta los ciclos de un grafo.
    
    conexiones -- matriz de conexiones del grafo (dirigido)
    '''
    return corta_ciclos_aux(conexiones, [0], 0, 1)          

def graph_metrics(g):
    '''
    Caracteriza un grafo devolviendo:
    ratio de conexiones, densidad del grafo, coeficiente de clustering medio 
    y coeficiente medio de camino corto entre pares.
    '''
    ratio_conexion = len(g.edges()) / len(g.nodes())
    densidad = nx.density(g)
    avg_clus = nx.average_clustering(g.to_undirected())
    
    try:
        avg_shrt = nx.average_shortest_path_length(g)
    except nx.NetworkXError:
        i = 0
        avg_shrt = 0
        for g_aux in nx.connected_component_subgraphs(g.to_undirected()):
            if (len(g_aux.nodes())>1):
                avg_shrt += nx.average_shortest_path_length(g_aux)
                i += 1
            
        avg_shrt /= i
            
    return ratio_conexion, densidad, avg_clus, avg_shrt

  
#######################################################

#runfile("./AnalyzeModel.py", "None")
    
if __name__ == '__main__':
    tipo_compresion = 0
    barajeo = None
    umbral_usado = 4
    
    if sys.argv[1] == '-t':
        isings, fits = train_ising(comprimir=tipo_compresion, gusanos = np.arange(0,1), umbral = umbral_usado, filename = "ising_filtrado.dat", temperatura = 1)
        
    else:
        isings, fits = restore_ising()

    entropias_calc = UmbralCalc.entropia_temperatura(isings[0])
    mejor_punto, valor = derivada_maxima_aproximada(np.arange(0,3,0.1), entropias_calc)
    (neural_activation,behavior)=worm.get_neural_activation(0)
    neural_activation = compresion(neural_activation, behavior, tipo_compresion)
    umbralizadas = umbralizar(neural_activation,umbral_usado)
    
    plt.figure()
    plt.plot(np.arange(0,1.5,0.1), entropias_calc[0:15])
    plt.plot(mejor_punto, valor,'ro')
            
    funcion, maximo, muestras, escala = aproximacion_sigmoidal(np.arange(0,1.5,0.1), entropias_calc[0:15], montecarlo=15)
    y = UmbralCalc.entropia(UmbralCalc.cuenta_estado(umbralizadas))/escala
    x = inversa_sigmoidal(y,*funcion)
    plt.plot(x,y*escala,'bo')
    print("Nuestro gusano es de listo: " + "{:.2f}".format(puntuar(y, maximo, funcion)[0]) + "/10")
    
    
