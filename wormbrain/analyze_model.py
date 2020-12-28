# -*- coding: utf-8 -*-
"""
@author: Javier Fumanal Idocin
"""
#!/usr/bin/env python
import sys
from . import worm
from . import entropy_metrics

from . import ising as isng
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
#import plotly.plotly as py
from . import math_tools as mt


from kinetic_ising import ising, bool2int
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
#from red_clasificador import entrenar_clasificador, process_labels
from .ising_recovery import save_isings, restore_ising, mandar_aviso_correo
from random import random
from plotly.graph_objs import Figure, Scatter, Line, Marker, Layout, Data, XAxis, YAxis
from timeit import default_timer as timer

#######################################################
#Parametros del programa
#######################################################
error=1E-3
variabilidad = 0.7
_barajeo = None
_ratio_nodo_arco = 11.14

##################################################
# Funciones
##################################################
def puntuar(resultado, maximo, parametros):
    '''
    Puntua del 0 al 10 como de cerca esta un valor de una sigmoidal de su
    punto de criticalidad.

    resultado -- valor a puntuar
    maximo -- valor donde la derivada de la sigmoidal es maxima
    parametros -- valores de ajuste de la sigmoidal (x0 y k)
    '''
    aux = mt.derivada_sigmoidal(mt.inversa_sigmoidal(resultado,*parametros),*parametros) / mt.derivada_sigmoidal(maximo,*parametros) * 10
    if aux > 10:
        aux = 10 - aux%10

    return aux

def capacidad_calorifica(modelo, rango=np.arange(-1,1.1,0.1), sigmoidal=False, normalizar=True, verboso = True):
    '''
    Devuelve las distintas capacidades calorificas del modelo para diferentes temperaturas

    modelo -- ising a estudiar
    rango -- rango de temperaturas a estudiar, de modo que T = 10**rango[n] (Exponentes elevados a 10)
    sigmoidal -- calcula la capacidad calorifica a partir de la derivdada de la sigmoidal si true
    '''
    res = np.zeros(len(rango))
    ind = 0
    aux = modelo.T

    if not sigmoidal:
        for i in rango:
            modelo.T = 10**i
            res[ind] = entropy_metrics.cap_calorifica(modelo)
            ind+=1

            if int(10**i) == aux:
                modelo_t = res[ind-1]
        modelo.T = aux
    else:
        maximo, funcion, escala = punto_criticalidad(modelo,True,rango=rango,verboso=False)
        for i in rango:
            res[ind] = mt.derivada_sigmoidal(i,*funcion)
            ind += 1

            if int(10**i) == aux:
                modelo_t = res[ind-1]

    if verboso:
        print("El modelo tiene un " + str(int(modelo_t/np.max(res)*100)) + "% del maximo de cap.calorifica")
        if normalizar:
            plt.ylabel("Capacidad calorifica normalizada")
        else:
            plt.ylabel("Capacidad calorifica")
        plt.xlabel("Temperatura")
        plt.plot(rango, res)

    if normalizar:
        return res/modelo.size
    else:
        return res

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
    global _barajeo
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
        plt.xlabel('Variabilidad acumulada')
        plt.ylabel('Variabilidad añadida')
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
        if _barajeo is None:
            _barajeo = np.arange(0,neural_activation.shape[1])
            np.random.shuffle(_barajeo)

        neural_activation = neural_activation[:,_barajeo[0:comprimir]]

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
    muestras = umbralizar(neural_activation[_barajeo],5)
    muestras_l = behavior[_barajeo]
    corte = int(porcentaje_train*muestras.shape[0])
    entrenar_clasificador(muestras[0:corte,:], muestras[corte+1:,:],process_labels(muestras_l[0:corte]),  process_labels(muestras_l[corte+1:]), filename =str(gusano)+'_gusano.json')


def train_ising(data_sets, kinetic=True, comprimir = 0, umbral = 0.17, aviso_email = False, filename = 'filename_ising.obj', temperatura = 1, tiempo = 1, alfa = 0.1, correo = "None"):
    '''
    Entrena un modelo de ising para cada uno de los gusanos dados.
    Los escribe en un fichero, ademas de devolverlos como resultado.
    Usa numba para optimizar el codigo.

    data_sets -- array con los conjuntos de datos a entrenar
    kinetic -- Si True, usara el modelo de Ising cinetico. (Suele ser mejor)
    comprimir -- indica el tipo de compresion a utilizar para la
                    dimensionalidad de las neuronas. (Consultar: compresion())
    umbral -- umbral a utilizar. Parametro funciona igual que para: umbralizar()
    aviso_email -- si True, avisara por correo electronico cuando cada gusano
                    termine de entrenar
    temperatura -- temperatura a la que poner a funcionar el sistema
    '''
    isings = [ising(1)]
    fits = [0.0]
    gusanos = len(data_sets)
    for gusano in range(gusanos):
        ##Cogemos los datos del gusano
        neural_activation = data_sets[gusano]

        ##Calculamos la dimension del array de las neuronas.
        size = neural_activation.shape[1] #Numero de dimensiones
        T = neural_activation.shape[0] #Numero de muestras

        activaciones = umbralizar(neural_activation, umbral)

        ##Calculamos la media y la covarianza de cada neurona
        sample = np.zeros(T)
        for i in range(T):
           sample[i] = (bool2int(activaciones[i,:]))

        m1, D1 = mt.calcMeanCov(sample, booleans = False, tiempo_=tiempo, size = size)
        start = timer()

        if (kinetic):
            y=ising(size)
            y.T = temperatura
            y.independent_model(m1)
            fit=y.inverse(m1,D1,error,sample, u=alfa)
        else:
           y=isng.ising(size)
           y.T = temperatura
           fit=y.inverse_exact(m1,D1,error)

        isings.append(y)
        fits.append(fit)

        if aviso_email:
            mandar_aviso_correo(str(gusano+1), correo)
        end = timer()
        diferencia = end-start
        minutos = diferencia /60
        segundos = diferencia%60
        horas = minutos/60
        minutos = minutos%60
        dias = horas%24
        horas = horas /24
        print("Terminado un entrenamiento: " + str(int(dias)) + " dias " + str(int(horas)) + " horas " + str(int(minutos)) + " minutos " + "{:.2f}".format(segundos) + " segundos")

    print("Escribiendo en fichero... ")
    save_isings(isings[1:], fits[1:], filename)

    print("Entrenamientos finalizados. Todo correcto")
    return isings[1:], fits[1:]


def punto_criticalidad(modelo, tr=True, rango=np.arange(-1,1.1,0.1), montecarlo_=21, verboso=True):
    '''
    Muestra por pantalla el punto de criticalidad del sistema ising dado.
    La funcion sigmoidal se calcula por defecto y como maximo, con 15 temperaturas aleatorias.

    ising -- sistema ising entrenado
    rango -- rango de temperaturas a estudiar.
    montecarlo -- numero de muestras a utilizar para aproximar la funcion por montecarlo.
    '''
    entropias_calc = entropy_metrics.entropia_temperatura(modelo, trans=tr)
    funcion, maximo, muestras, escala = mt.aproximacion_sigmoidal(rango, entropias_calc, montecarlo=min(montecarlo_,len(rango)), verboso=verboso)

    if not tr:
       y = entropy_metrics.entropia(entropy_metrics.cuenta_estado(umbralizadas))/escala
       x = mt.inversa_sigmoidal(y,*funcion)
    else:
       x = np.log10(modelo.T)
       y = mt.sigmoidal(x,*funcion)

    if verboso:
        plt.plot(x,y*escala,'bo')

    return maximo, funcion, escala

def _buscar_estable(model, max_intentos=1000):
    '''
    Termina cuando el sistema ising ha llegado a un punto estable.
    Devuelve de forma aproximada el numero de iteraciones que le ha costado
    llegar.

    ising -- sistema ising a medir
    iteraciones -- x numero de muestras en cada remesa de muestras
    max_intentos -- numero de remesas maximo a generar
    '''
    intentos = 0
    medias = np.zeros(model.size)

    while max_intentos >= intentos:
        intentos += 1
        model.GlauberStep()
        medias += model.s

    return np.abs(medias/intentos)


def calculo_magnetismo(ising, precision = 500, rango_estudio = 10**np.arange(-1,1.1,0.1), verboso = True):
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
    T_original =  ising.T
    medias = np.zeros(len(rango_estudio))

    for i in np.arange(0,rango_estudio.size):
        if verboso:
            print("Con T igual a: " + str(rango_estudio[i]))
        ising.T = rango_estudio[i]
        medias[i] = (np.sum(_buscar_estable(ising, precision)))

    if verboso:
        plt.figure()
        plt.semilogx(rango_estudio, medias)
        plt.ylabel('Indice ')
        plt.xlabel('Temperatura')
        plt.xscale('log')

    ising.T = T_original
    return medias



def _largest_indices(ary, n):
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
    global _ratio_nodo_arco

    conexiones = []
    nodos_a_coger = int(muestras[0].shape[0] * _ratio_nodo_arco)

    for matrix in muestras:
        for i in np.arange(0,matrix.shape[0]):
            for j in np.arange(0,matrix.shape[1]):
                if i==j:
                    matrix[i,j] = 0
        if not ratio_real:
            umbral = min(np.mean(matrix), np.median(matrix))
            detectadas = np.less_equal(matrix, umbral)
        else:
            indexes = _largest_indices(matrix, nodos_a_coger)
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

def construir_grafo(conexiones):
    '''
    Construye un grafo dirigido en base a una matriz de conexiones.

    conexiones -- matrix de conexiones [i,j] implica que i -> j
    '''
    g=nx.DiGraph()
    for i in range(conexiones.shape[0]):
        g.add_node(i, pos = [random(),random()])

    for i in range(conexiones.shape[0]):
        for j in range(conexiones.shape[1]):
            if (i != j) and (conexiones[i,j]):
                g.add_edge(i,j)

    return g

def reconstruir_red(muestra, ratio_real = True, fiabilidad = 1, cut_ciclos = True, verboso=True):
    '''
    A partir de una serie de muestras de transmision de entropia entre neuronas
    reconstruye una aproximacion de las conexiones reales entre neuronas.
    Devuelve un grafo dirigido y la matriz de conexiones.

    muestra -- entropia de cada conexion por pares en distintos tiempos.
    fiabilidad -- nos quedaremos
    verboso -- muestra por pantalla las conexiones obtenidas
    '''
    global _ratio_nodo_arco

    fiabilidades = busca_conexiones(muestra, ratio_real)

    if not ratio_real:
        fiabilidades = fiabilidades*1.0 / len(muestra)
        conexiones = np.greater_equal(fiabilidades, fiabilidad)
    else:
        nodos_a_coger = int(muestra[0].shape[0] * _ratio_nodo_arco)
        indexes = _largest_indices(fiabilidades, nodos_a_coger)
        conexiones = np.zeros(fiabilidades.shape)
        conexiones[indexes] = 1

    if cut_ciclos:
        conexiones = corta_ciclos(conexiones)

    if verboso:
        plt.imshow(conexiones)

    g = construir_grafo(conexiones)

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
    for n in pos:
        x,y=pos[n]
        d=(x-0.5)**2+(y-0.5)**2
        if d<dmin:
            dmin=d

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

def _corta_ciclos_aux(conexiones, visitados, inicio, final):
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
                    conexiones = _corta_ciclos_aux(conexiones, visitados, j, j+1)

        visitados = inicial

    return conexiones

def corta_ciclos(conexiones):
    '''
    Corta los ciclos de un grafo.

    conexiones -- matriz de conexiones del grafo (dirigido)
    '''
    return _corta_ciclos_aux(conexiones, [0], 0, 1)

def graph_metrics(g):
    '''
    Caracteriza un grafo devolviendo:
    ratio de conexiones, densidad del grafo, coeficiente de clustering medio,
    porcentaje de conexiones posibles y coeficiente medio de camino corto entre pares.
    '''
    ratio_conexion = len(g.edges()) / len(g.nodes())
    densidad = nx.density(g)
    avg_clus = nx.average_clustering(g.to_undirected())
    posibles = len(g.edges()) / len(g.nodes())**2
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

    return ratio_conexion, densidad, posibles, avg_clus, avg_shrt

def validate_model(muestra, model=None, entrenar = True, verboso = True, estado_inicial = 0, tiempo_ = 1, tam=None):
    '''
    Devuelve las medias y covarianzas de una muestra y de una muestra generada
    a partir de un modelo.

    muestra -- muestra a analizar
    entrenar -- si True, creara y entrenara un nuevo modelo para la nuestra
    verboso -- si True, muestra por pantalla las imagenes
    estado_inicial--
        si 0 -> estado inicial aleatorio
        si 1 -> estado inicial aleatorio + tiempo de simulacion previo a la medida para evitar ruido incial
        si 2 -> estado inicial uno de la muestra
    tiempo_ -> X -> P(t+X)
    '''
    if model is None:
        if entrenar:
           isings, fits = train_ising(gusanos = np.arange(0,1), umbral = 4, tiempo = tiempo_)
        else:
           isings, fits = restore_ising()

        model = isings[0]

    if estado_inicial == 1:
        model.generate_sample(tam)

    elif estado_inicial == 2:
        model.s = muestra[0]*2-1

    m0, D0 = mt.calcMeanCov(muestra)
    muestra_artificial = model.generate_sample(tam)
    m1, D1 = mt.calcMeanCov(muestra_artificial, booleans = False, size=muestra.shape[1], tiempo_=tiempo_)

    if verboso:
        plt.figure()
        plt.plot(m0, 'ro')
        plt.xlabel("Neurona")
        plt.ylabel("Medias originales")
        plt.figure()
        plt.plot(m1, 'ro')
        plt.xlabel("Neurona")
        plt.ylabel("Medias modelo")
        mt.color_bar(D0, "Correlaciones originales")
        mt.color_bar(D1, "Correlaciones del modelo")

    return m0,D0, m1,D1


#######################################################

#runfile("./analyze_model.py", "None")

if __name__ == '__main__':
    #Si como compresion se escoge un numero de neuronas aleatorio:
    #   Si barajeo == None: se escogen y barajeo para a indicar las elegidas.
    #   Si barajero != None, se escogen las indicadas en barajeo
    '''
    _barajeo = None

    tipo_compresion = 7
    umbral_usado = 4
    tiempo_ = 1
    tr = True
    gusano=0
    umbralizadas = worm.quick_load(gusano, tipo_compresion, umbral_usado)

    if sys.argv[1] == '-t':
        data = [umbralizadas]
        isings, fits = train_ising(data, comprimir=tipo_compresion , umbral = umbral_usado, filename = "ising_filtrado_t" + str(tiempo_)+"_"+str(tipo_compresion)+".dat", temperatura = 1, tiempo = tiempo_)

    else:
        if sys.argv[1] == "None":
            isings, fits = restore_ising("ising_filtrado_t"+str(tiempo_)+"_"+str(tipo_compresion)+".dat")
            isings[0].T=1
        else:
            isings, fits = restore_ising(sys.argv[1])
            isings[0].T=1
    entropias_calc = entropy_metrics.entropia_temperatura(isings[0], trans=tr)

    #Para sacar la aproximacion
    #mejor_punto, valor = mt.derivada_maxima_aproximada(np.arange(-1,1.1,0.1), entropias_calc)
    #plt.figure()
    #plt.plot(np.arange(-1,1.1,0.1), entropias_calc[0:21])
    #plt.plot(mejor_punto, valor,'ro')

    funcion, maximo, muestras, escala = mt.aproximacion_sigmoidal(np.arange(-1,1.1,0.1), entropias_calc, montecarlo=21)
    if not tr:
        y = entropy_metrics.entropia(entropy_metrics.cuenta_estado(umbralizadas))/escala
        x = 10**mt.inversa_sigmoidal(y,*funcion)
        print("Nuestro gusano es de listo: " + "{:.2f}".format(puntuar(y, maximo, funcion)[0]) + "/10")
    else:
        x = np.log10(isings[0].T)
        y = mt.sigmoidal(x,*funcion)

    plt.semilogx(10**x,y*escala,'bo')
    '''
    import pandas as pd

    for subj in range(20):
        subject = str(subj+1)

        tests = [np.array(pd.read_csv('/home/fcojavier.fernandez/computational_brain/spikes/s'+ subject + '/'+str(test+1)+'.csv', index_col=False, header=None)).T for test in range(20)]
        ans_ising = train_ising(tests, umbral=0, kinetic=False, filename=subject + '_isings.npy')






