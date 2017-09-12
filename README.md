# Wormbrain
Wormbrain es un libreria que ofrece herramientas para trabajar con datos de actividad neuronal.

Este repositorio es un fork del repositorio ya existente de Miguel Aguilera: https://github.com/MiguelAguilera/wormbrain


## Caracteristicas:

  - Discretizacion de la actividad neuronal.
  - Inferencia de modelos de Ising y funciones para comprobar su fiabilidad.
  - Metricas relacionadas con el estudio de la entropia.
  - Reconstruccion aproximada de la red neuronal original.
  - Dibujo de grafos en plotly.
  - Herramientas para el estudio de la criticidad de la muestra/modelo.
  - Herramientas para facilitar la visualizaciÃ³n y carga/guardado de muestras y resultados.
  - Clasificador para asociar actividad neuronal con comportamientos/estados determinados.

## Instalacion

Wormbrain requiere de Python 3.X para funcionar.
Wormbrain requiere de los siguientes paquetes para funcionar:

 - Matplotlib
 - Networkx
 - Plotly
 - Sklearn
 - Numba
 - Numpy
 - Numexpr

Ademas, se necesita del siguiente repositorio de Miguel Aguilera: https://github.com/MiguelAguilera/Python-Entropy-Tools
En caso de utilizar el fichero setup.py para instalar Wormbrain, hay que introducir los ficheros contenidos en este repositorio en la carpeta raiz de Wormbrain.

## Estructura
Los principales ficheros son:

- analyze_model: contiene todas las funciones de anÃ¡lisis del modelo, ya sea para buscar el punto critico, o posibles conexiones de la red neuronal original de partir de los datos. Si se ejecuta, carga un modelo de Ising ya guardado y calcula el punto critico del sistema en comparacion con la muestra con la que fue calculado.
- kinetic_ising: contiene todas las funciones para crear y entrenar un modelo Ising cinetico a partir de un conjunto de datos, asi como generar muestras a partir de el.
- entropy_metrics: contiene funciones para realizar medidas de entropia y otras relacionadas con la misma, como la capacidad calorifica del modelo.
- ising_recovery: contiene funciones para guardar y cargar de ficheros modelos, muestras y resultados.
- math_utils: contiene funciones para aproximar datos reales bien a una función sigmoidal, bien a una distribución de Zipf, además de para facilitar la visualización de datos. 
- red_clasificador: contiene las funciones para crear y entrenar una red neuronal con Keras como clasificador.
- worm: contiene funciones de QQL para trabajar con el set de datos de activacionn neuronal del gusano C.elegans.




<p align="center">
  <img src="https://0uunvw-ch3302.files.1drv.com/y4m3kkkGGV6HE_P4HmkoaNqXf6PqEf_aPQsn0yn0TWUUoYfTUuIO-psNG7bIXysG-vp-HdMac7nFOgV9pKJ0eWcXY2lXiGCxsGwEEKliGi9xHiJaoFvouYywAbF4f4AlrLNw-7BKMylsM9VNRzIJPdS6b6va_vvc5juTtK9b2KPZhZ4X-XIe-VkIo9UbMk8_gMAjyjtvE7PDkVeq6-aO4alIQ?width=256&height=231&cropmode=none"/>
</p>
 
