# -*- coding: utf-8 -*-
"""
@author: Javier Fumanal Idocin
"""
import pickle 
import matplotlib as mpl
import numpy as np

from PIL import Image


def save_isings(isings, fits, filename = 'filename_ising.obj'):
    '''
    Guarda los ising en un fichero junto con sus errores.
    
    isings - isings a guardar
    fits - medidas de error
    filename - nombre del fichero a utilizar. (Tiene valor por defecto)
    '''
    
    file_write = open('./Modelos/' + filename, 'wb+')
    pickle.dump(isings, file_write)
    pickle.dump(fits, file_write)
    file_write.close()

def restore_ising(filename = 'filename_ising_no_filtro.obj'):
    '''
    Lee y devuelve los isings guardados junto con sus medidas de error
    
    filename -- Nombre del fichero donde estan contenidos. Por defecto es
                es el mismo que para guardarlos.
    '''
    file_read = open('./Modelos/' + filename, 'rb')
    isings = pickle.load(file_read)
    fits = pickle.load(file_read)
    file_read.close()
    return isings, fits

def generate_muestra(ising, filtro, size = 10000):
    '''
    Genera una muestra del ising dado, y la guarda en un fichero
    
    ising -- modelo a utilizar para generar la muestra
    filtro -- indica si se ha usado filtro de paso alto o no
    size -- numero de muestras a generar
    '''
    muestra = ising.generate_sample(size)
    
    file_write = open("./Muestras/muestra_tamano_" + str(size) + "_neuronas_" + str(ising.size) + "_temperatura_" + str(ising.T) + "_filtered_" + str(filtro), 'wb+')
    pickle.dump(muestra, file_write)
    file_write.close()
    
    return muestra

def load_muestra(size, neuronas, temperatura, filtro):
    '''
    Devuelve una muestra cargada en un fichero dado.
    
    size -- tamano de la muestra
    neuronas -- numero de las neuronas
    temperatura -- indica la temperatura del ising a funcionar
    filtro -- indica si se ha usado filtro o no
    '''
    file_write = open("./Muestras/muestra_tamano_" + size + "_neuronas_" + neuronas + "_temperatura_" + temperatura + "_filtered_" + filtro, 'rb')
    muestra = pickle.load(file_write)
    file_write.close()
    return muestra

def save_results(variable, nombre):
    '''
    Guarda todas las variables del workspace en un fichero con la fecha actual
    
    variable -- variable a guardar
    nombre -- nombre de la variable (asi se habra guardado el fichero)
    '''
    file_write = open("./Resultados/" + str(nombre), 'wb+')
   
    pickle.dump(variable, file_write)
        
    file_write.close()

def load_results(nombre):
    '''
    Carga en una variable lo contenido en un fichero.
    
    nombre -- fecha y nombre de la variable guardada con anterioridad.
    '''
    file_write = open("./Resultados/" + nombre, 'rb')
    vari = pickle.load(file_write)
    file_write.close()
    
    return vari

def save_image(data, nombre):
    '''
    Guarda un array de numpy como una imagen con el mapa de colores magma.
    En formato PNG.
    
    data-- array a guardar
    nombre -- nombre con el que guardar la imagen (sin extension)
    '''
    rescaled = (255.0 / data.max() * (data - data.min())).astype(np.uint8)
    img_src = Image.fromarray(rescaled).resize([400,400])
    cm_hot = mpl.cm.get_cmap('magma')
    im = np.array(img_src)
    im = cm_hot(im)
    im = np.uint8(im * 255)
    im = Image.fromarray(im)
    
    im.save("./Resultados/" + nombre + ".png")
