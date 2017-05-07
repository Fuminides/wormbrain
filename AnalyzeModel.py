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

from kinetic_ising import ising, bitfield, bool2int
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from numba import jit
from scipy.optimize import curve_fit
from RedClasificador import entrenar_clasificador
from IsingRecovery import save_isings, restore_ising



sys.path.insert(0, '..')

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
    '''
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

    return neural_activation


def umbralizar(neural_activation, umbral, verboso=False, filtrado = True):
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
    
    if filtrado:
        b, a = scipy.signal.butter(8, 0.01,btype='highpass')
        neural_activation = scipy.signal.filtfilt(b, a, neural_activation)
        
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
    
def crear_clasificador(gusano, filename = 'defecto', umbral = 5):
    '''
    Crea y guarda en un fichero un clasificador que detecta el comportamiento
    del gusano a partir de su actividad neuronal
    '''
    (neural_activation,behavior)=worm.get_neural_activation(gusano)
    behavior = np.maximum(behavior, np.zeros(behavior.size))
    
    porcentaje_train = 0.8
    
    barajeo = np.arange(behavior.size)
    np.random.shuffle(np.arange(behavior.size))
    muestras = umbralizar(neural_activation[barajeo],5)
    muestras_l = behavior[barajeo]
    corte = int(porcentaje_train*muestras.shape[0])
    entrenar_clasificador(muestras[0:corte], muestras[corte+1:], muestras_l[0:corte], muestras_l[corte+1:], filename =gusano+'_gusano.json')

    
@jit
def train_ising(kinectic=True, comprimir = 0, umbral = 0.17, aviso_email = False, gusanos = np.arange(0,5), filename = 'filename_ising.obj', temperatura = 1):
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
    '''
    isings = []
    fits = []
    
    for gusano in gusanos:
        ##Cogemos los datos del gusano
        (neural_activation,behavior)=worm.get_neural_activation(gusano)
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
    save_isings(isings, fits, filename)
    
    print("Entrenamientos finalizados. Todo correcto")
    return isings, fits
    
    
def punto_criticalidad(ising, tipo_compresion, gusano, montecarlo=15):
    '''
    Muestra por pantalla el punto de criticalidad del sistema ising dado.
    La funcion sigmoidal se calcula por defecto y como maximo, con 15 temperaturas aleatorias.
    
    ising -- sistema ising entrenado
    tipo_compresion -- sistema de reduccion de dimensionalidad a utilizar para las neuronas.
                (Usar el mismo que el usado para entrenar el ising)
    gusano -- gusano con el que comparar la entropia. (Usar el mismo que el entrenado con ising)
    '''
    entropias_calc = UmbralCalc.entropia_temperatura(ising, tipo_compresion)
    (neural_activation,behavior)=worm.get_neural_activation(0)
    neural_activation = compresion(neural_activation, behavior, tipo_compresion)
    
    plt.figure()
    plt.plot(np.arange(0,1.5,0.1), entropias_calc[0:15])
    plt.plot(mejor_punto, valor,'ro')
    
    resultado = UmbralCalc.entropia_muestra(umbralizar(neural_activation,5), 2)
    plt.axhline(y=resultado, color='r', linestyle='-')
    
    funcion, maximo, muestras = aproximacion_sigmoidal(np.arange(0,1.5,0.1), entropias_calc[0:15], montecarlo=15)
    
def distribucion_probabilidad_estados(muestras, verboso = False):
    '''
    Devuelve ordenadas las probabilidades de cada estado de la muestra de mayor a menor.
    
    muestras -- array de muestras
    entero -- indica si las muestras van en forma de enteros o de array de bools
    verboso -- ensenya una grafica con las probabilidades (x logaritmo)
    '''
    ocurrencias = list(UmbralCalc.cuenta_estado_int(muestras).values())
        
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
    
#######################################################
#Parametros del programa
#######################################################
gusano=1
error=1E-3
variabilidad = 0.85

umbral = 3
comprimir = 1
#######################################################

#runfile("./AnalyzeModel.py", "None")
if __name__ == '__main__':
    tipo_compresion = 0
    umbral_usado = 6
    
    if sys.argv[1] == '-t':
        isings, fits = train_ising(comprimir=tipo_compresion, gusanos = np.arange(0,1), umbral = umbral_usado, filename = sys.argv[1], temperatura = 0.4)
        
    else:
        isings, fits = restore_ising()
        isings[0].T = 1

    entropias_calc = UmbralCalc.entropia_temperatura(isings[0])
    mejor_punto, valor = derivada_maxima_aproximada(np.arange(0,3,0.1), entropias_calc)
    (neural_activation,behavior)=worm.get_neural_activation(0)
    neural_activation = compresion(neural_activation, behavior, tipo_compresion)
    umbralizadas = umbralizar(neural_activation,umbral_usado)
    
    plt.figure()
    plt.plot(np.arange(0,1.5,0.1), entropias_calc[0:15])
    plt.plot(mejor_punto, valor,'ro')
            
    funcion, maximo, muestras, escala = aproximacion_sigmoidal(np.arange(0,1.5,0.1), entropias_calc[0:15], montecarlo=15)
    y = UmbralCalc.entropia(UmbralCalc.cuenta_transiciones(umbralizadas))/escala
    x = inversa_sigmoidal(y,*funcion)
    plt.plot(x,y*escala,'bo')
    print("Nuestro gusano es de listo: " + "{:.2f}".format(puntuar(y, maximo, funcion)[0]) + "/10")
    
    
