# -*- coding: utf-8 -*-
"""
@author: Javier Fumanal Idocin
"""
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json

##### VARIABLES GLOBALES ######
max_layers = 5
numero_behaviours = 3
###############################


##################################################
# Funciones
##################################################

def aciertos(real, yhat):
	'''
	Comprueba uno a uno los valores reales con las predcciones y pone en cada
    valor del array un 0 si el termino i-esimo no fue acertado correctamente,
    o un 1 si si que lo fue.
	'''
	resultado = []
	for x in range(len(real)):
		add = 0
		if real[x] == yhat[x]:
			add = 1

		resultado.append(add)

	return resultado

def busca1(vector):
    for i in np.arange(0,vector.size):
        if vector[i] == 1:
            return i
        
def indices_maximos_vector(matriz):
	'''
	Devuelve el indice con valor maximo de un vector
	'''
	resultados = [0] * len(matriz)

	for i in range(len(matriz)):
		resultados[i] = matriz[i].tolist().index(max(matriz[i])) + 1

	return resultados

def process_labels(labels_originales):
    '''
    
    '''
    tam = labels_originales.size
    nuevas_labels = np.zeros([tam,numero_behaviours])
    for i in np.arange(0,tam):
        nuevas_labels[i,labels_originales[int(i)]] = 1
        
    return nuevas_labels

def entrenar_clasificador(train, test, l_train, l_test, filename = 'y_model_architecture.json'):
    '''
    Entrena una red neuronal a partir de unos datos de train y de test.
    (Genera automaticamente los de validacion a partir del train)
    
    Comprueba como se comporta la red con distinto numero de capas y numeros de
    neuronas hasta dar con la que mejor funciona. El numero maximo de layers depende
    de la variable global @max_layers.
    
    Guarda el clasficiador entrenado en un fichero de no tocar.
    
    train -- datos de entrenamiento. La matriz debe estar compuesta por
                vectores de 0s y 1s.
    test -- datos de test. Mismo formato que train.
    l_train -- labels de las muestras de entrenamiento. Deben ser un numero comprendido
    entre 0 y la variable global @numero_behaviours. (Por defecto a 3, pero 
    se debe de adecuar al data set a entrenar)
    filename - nombre del fichero JSON donde guardar la red neuronal.
    '''
    dimensiones = train.shape[1]
    mejores_tamanos = [0] * (max_layers + 1)
    mejor_error_global = 1
    test_labels = indices_maximos_vector(l_test)
    
    for layer in np.arange(0,max_layers):
        print("Entrenando con " + str(layer+1) + " layers")
        mejor_error_layer = 1
        
        for neuronas in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
            model = Sequential()
            model.add(Dense(output_dim = dimensiones, input_dim = dimensiones, activation='linear'))
            
            for layers_pasados in np.arange(0,layer):
                if layers_pasados != 0:
                    model.add(Dense(output_dim = mejores_tamanos[layers_pasados], activation='tanh'))
#            
           # model.add(Dense(output_dim = dimensiones, activation = 'tanh'))
            model.add(Dense(output_dim = numero_behaviours, activation = 'softplus'))
            model.compile(optimizer = 'adadelta', loss = 'categorical_crossentropy', metrics = ['accuracy'])
            model.fit(train, l_train, nb_epoch=5, verbose = 0, shuffle = True)

            yhat = model.predict(test)
            yhat = indices_maximos_vector(l_test)
            exitos = aciertos(test_labels, yhat)
            error = 1 - (sum(exitos)/len(yhat))

            print ("Layers ocultos: " + str(layer) + " Neuronas: " + str(neuronas) + " Error: " + str(error * 100) + "%")
            if mejor_error_layer > error:
                mejor_error_layer = error
                #print ("Layer: " + str(layer) + " Len: " + str(len(mejores_neuronas)))
                mejores_tamanos[layer] = neuronas

            if error < mejor_error_global:
                print("Mejoramos la red")
                mejor_error_global = error

            #Guardar la red
            json_string = model.to_json()
            open('./NoTocar/'+ filename, 'w').write(json_string)
            model.save_weights('./NoTocar/my_model_weights.h5', overwrite=True)
    
def cargar_clasificador():
    '''
    Carga un clasificador previamente entrenado de la carpeta NoTocar.
    '''
    model = model_from_json(open('./NoTocar/my_model_architecture.json').read())
    model.load_weights('./NoTocar/my_model_weights.h5')
    model.compile(optimizer='adadelta', loss='categorical_crossentropy')
    
    return model

def utilizar_clasificador(model, muestra):
    '''
    Dado un clasificador y una muestra, devuelve la predicción de un clasificador
    para esa muestra.
    '''
    yhat = model.predict([muestra])
    yhat = indices_maximos_vector(yhat)
    
    if (yhat[0] == 0):
        return -1
    
    return yhat[0]
    
    
            