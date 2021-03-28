[![Build Status](https://travis-ci.org/joemccann/dillinger.svg?branch=master)](https://travis-ci.org/joemccann/dillinger)

# deep AR

Paper deep AR de amazon, implementación a través de gluonts y MX-NET

Esta en desarrollo, este repo ...


### Instalar las librerías necesarias para trabajar con deepAR en gluonts
```sh
$ git clone https://github.com/matheus695p/deep-ar.git
$ cd deep-ar
$ pip install -r requirements.txt
```

Ej: a través de anaconda, tiene que ser en ambiente con python=3.6
```sh
$ conda create -n deepAR python=3.6 anaconda
$ conda activate deepAR
$ conda install pip
$ pip install pandas==1.0.5
$ pip install --upgrade mxnet==1.6
$ pip install gluonts
```


tree del proyecto

```sh
│   .gitignore
│   README.md
│   requirements.txt
├───codes
│   ├───electricity
│   │       main.py   ---> códicos para hacer predicción de demanda de electricicdad
│   │
│   └───manufacturing
│           main.py   ---> códicos para hacer predicción de demanda de skus, caso de manufactura
│           transformations.py   ---> códicos para transformación al formato de datos de gluonts
├───data
│       LD2011_2014.txt
│       manufacturing.csv
│       raw_manufacturing.csv
├───docs
├───documents
│       paper-amazon-deepAR.pdf ---> paper de deep autoregresive neural networks
│       paper-gluontsTS.pdf ---> paper de la implementación en gluonts
├───images  ---> resultados para los difererentes casos
│       caso_electricidad_{nombre}.png
│       caso_manufactura_{nombre}.png
├───results  ---> resultados para los difererentes casos
│   │   resultados_energia.csv  
│   │   resultados_manufactura.csv
│   ├───energy-model  ---> modelo de energia
│   │       input_transform.json
│   │       parameters.json
│   │       prediction_net-0000.params
│   │       prediction_net-network.json
│   │       type.txt
│   │       version.json
│   └───manufacturing-model  ---> modelo de manufactura
│           input_transform.json
│           parameters.json
│           prediction_net-0000.params
│           prediction_net-network.json
│           type.txt
│           version.json
└───src
    │   module.py  ---> módulo de funciones
    └───__init__.py
```

## Resultados

Los scripts utilizados para hacer las pruebas se pueden encontrar en 
```sh
├───codes
│   ├───electricity
│   │       main.py   ---> códicos para hacer predicción de demanda de electricicdad
│   └───manufacturing
│           main.py   ---> códicos para hacer predicción de demanda de skus, caso de manufactura
│           transformations.py   ---> códicos para transformación al formato de datos de gluonts
│           stationary_tests.py   ---> códicos para verificar estacionaridad de las series de tiempo
│           make_stationary.py   ---> códicos para hacer estacionarias las series de tiempo
```


### Caso de manufactura:

Los datos manejados son de supply, los que equivalen a la demanda de diferentes articulos en el tiempo de una empresa manufacturera, para probar deep AR se eligen los sku's con mayor cantidad de ventas realizadas en el tiempo, de tal manera de abarcar el 80/20 de la producción. En el caso de manufactura es más complejo el preprocesamiento de las series de tiempo, dado que a diferencia del caso academico de energia, es necesario: (1) Hacer tests estadísticos de estacionaridad, (2) En el caso de que no sean estacionarias, por resultado del test, es necesario aplicar técnicas, para llevarlas a ser estacionarias, con el fin de que los módelos de forecasting, tengan el trabajo más fácil. (3) Entrenar modelos con arquitecturas deep AR (4) finetuning a los módelos.

#### Estacionaridad:

Hay algunas nociones más detalladas de estacionariedad que puede encontrar si profundiza en este tema. Son:
Son:
* Proceso estacionario (stationary process): proceso que genera una serie estacionaria de observaciones.
* Modelo estacionario (stationary model): un modelo que describe una serie estacionaria de observaciones.
* Tendencia estacionaria (trend c): una serie de tiempo que no muestra una tendencia.
* Estacional por periodos (seasonal stationarity): una serie de tiempo que no exhibe estacionalidad.
* Estrictamente estacionario (strictly (stationary model)): una definición matemática de un proceso estacionario, específicamente que la distribución conjunta de observaciones es invariante al cambio de tiempo.

Podemos usar una prueba estadística para verificar si la diferencia entre dos muestras de variables aleatorias gaussianas es real o una casualidad estadística. Podríamos explorar pruebas de significación estadística, como la prueba t de Student. En esta parte del trabajo, se utiliza una prueba estadística diseñada para comentar explícitamente si una serie de tiempo univariante es estacionaria. El test se llama Augmented Dickey-Fuller.

Hay una serie de pruebas de raíz unitaria y Augmented Dickey-Fuller puede ser una de las más utilizadas. Utiliza un modelo autorregresivo y optimiza un criterio de información a través de múltiples valores de retardo (lags) diferentes. La hipótesis nula de la prueba es que la serie de tiempo se puede representar mediante una raíz unitaria, que no es estacionaria (tiene alguna estructura dependiente del tiempo). La hipótesis alternativa (que rechaza la hipótesis nula) es que la serie de tiempo es estacionaria.

* Hipótesis nula (H0): si no se rechaza, sugiere que la serie de tiempo tiene una raíz unitaria, lo que significa que no es estacionaria. Tiene alguna estructura dependiente del tiempo --> problemas.
* Hipótesis alternativa (H1): Se rechaza la hipótesis nula; sugiere que la serie de tiempo no tiene una raíz unitaria, lo que significa que es estacionaria. No tiene una estructura dependiente del tiempo.

Voy a interpretar este resultado utilizando el valor p de la prueba. Un valor p por debajo de un umbral (como 5% o 1%) sugiere que rechazamos la hipótesis nula (estacionaria); de lo contrario, un valor p por encima del umbral sugiere que no rechazamos la hipótesis nula (no estacionaria), este es el lo clásico en tests estadísticos.

* Valor p> 0.05: No se rechaza la hipótesis nula (H0), los datos no son estacionarios
* Valor de p <= 0.05: Rechaza la hipótesis nula (H0), los datos son estacionarios



