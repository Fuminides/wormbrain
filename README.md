# Wormbrain
Wormbrain es un librería que ofrece herramientas para trabajar con datos de actividad neuronal.

Este repositorio es un fork del repositorio ya existente de Miguel Aguilera: https://github.com/MiguelAguilera/wormbrain


## Características:

  - Discretización de la actividad neuronal.
  - Inferencia de modelos de Ising y funciones para comprobar su fiabilidad.
  - Métricas relacionadas con el estudio de la entropía.
  - Reconstruccion aproximada de la red neuronal original.
  - Dibujo de grafos en plotly.
  - Herramientas para el estudio de la criticidad de la muestra/modelo.
  - Herramientas para facilitar la visualización y carga/guardado de muestras y resultados.
  - Clasificador para asociar actividad neuronal con comportamientos/estados determinados.

## Instalación

Wormbrain requiere de Python 3.X para funcionar.
Wormbrain requiere de los siguientes paquetes para funcionar:

 - Matplotlib
 - Networkx
 - Plotly
 - Sklearn
 - Numba
 - Numpy

Además, se necesita del siguiente repositorio de Miguel Aguilera: https://github.com/MiguelAguilera/Python-Entropy-Tools
En caso de utilizar el fichero setup.py para instalar Wormbrain, hay que introducir los ficheros contenidos en este repositorio en la carpeta raíz de Wormbrain.

## Estructura
Los principales ficheros son:

- AnalyzeModel: contiene todas las funciones de análisis del modelo, ya sea para buscar el punto crítico, o posibles conexiones de la red neuronal original de partir de los datos. Además de poseer algunas funciones adicionales de QQL. Si se ejecuta, carga un modelo de Ising ya guardado y calcula el punto crítico del sistema en comparación con la muestra con la que fue calculado.
- KineticIsing: contiene todas las funciones para crear y entrenar un modelo Ising cinético a partir de un conjunto de datos, así como generar muestras a partir de él.
- UmbralCalc: contiene funciones para realizar medidas de entropía y otras relacionadas con la misma, principalmente pensadas para buscar un buen umbral con el que discretizar un conjunto de datos.
- IsingRecovery: contiene funciones para guardar y cargar de ficheros modelos, muestras y resultados.
- RedClasificador: contiene las funciones para crear y entrenar una red neuronal con Keras como clasificador.
- Worm: contiene funciones de QQL para trabajar con el set de datos de activación neuronal del gusano C.elegans.




<p align="center">
  <img src="https://0uunvw-ch3302.files.1drv.com/y4m3kkkGGV6HE_P4HmkoaNqXf6PqEf_aPQsn0yn0TWUUoYfTUuIO-psNG7bIXysG-vp-HdMac7nFOgV9pKJ0eWcXY2lXiGCxsGwEEKliGi9xHiJaoFvouYywAbF4f4AlrLNw-7BKMylsM9VNRzIJPdS6b6va_vvc5juTtK9b2KPZhZ4X-XIe-VkIo9UbMk8_gMAjyjtvE7PDkVeq6-aO4alIQ?width=256&height=231&cropmode=none"/>
</p>
 
