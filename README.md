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
$ pip install pytorchts==0.3.1
$ echo en mi caso necesitaba install cudnn para correr en la GPU
$ pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio===0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
$ pip install pandas==1.0.5
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
```


### Caso de manufactura:

Los datos manejados son de supply, los que equivalen a la demanda de diferentes articulos en el tiempo de una empresa manufacturera, para probar deep AR se eligen los sku's con mayor cantidad de ventas realizadas en el tiempo, de tal manera de abarcar el 80/20 de la producción. En el caso de manufactura es más complejo el preprocesamiento de las series de tiempo, dado que a diferencia del caso academico de energia, es necesario: (1) Hacer tests estadísticos de estacionaridad, (2) En el caso de que no sean estacionarias, por resultado del test, es necesario aplicar técnicas, para llevarlas a ser estacionarias, con el fin de que los módelos de forecasting, tengan el trabajo más fácil. (3) Entrenar modelos con arquitecturas deep AR (4) finetuning a los módelos.

#### Preprocessing:
Para el preprocesamiento de los datos, solo se trabajara en esta prueba de conceptos con bases de sku's, existen al rededor de 30.000 sku's diferentes, por lo que se hace imposible realizar modelo para cada sku, además las para ellos se extraen las bases de un sku y con ellas se trabaja.

* SKU = **METAL # 4.5**|11|540|DO|COMPANY|ZP48 ZAMAC
* BASE_SKU = **METAL # 4.5**

De esta forma se reduce a trabajar solamente con 19 productos, del total de sku's que son las base del 80/20 de la empresa en cuestión.
formato usado para el ingreso de los datos a deepAR:
```sh 
    ─── columnas: bases de sku
   │   
   │
filas: compras
realizadas de
esa base de
sku
```

#### feature engineering:

¿Cómo manejar los largos períodos sin demanda que no siguen un patrón específico?"

La respuesta esta pregunta es Análisis de demanda intermitente o Análisis de datos dispersos, esto pasa cuando existen "muchos ceros" en relación con el número de no ceros. El problema es que hay dos variables aleatorias, la primerisima es el tiempo entre eventos (a tu elección) y el tamaño esperado del evento. Si vemos gráficos de autocorrelación (acf) del conjunto completo de lecturas no tiene ningún sentido debido a que la secuencia de ceros realza falsamente el acf (no hay ningún patrón).
Hay un par de enfoques para resolver esto, en primera instancia nos quedaremos con el más fácil (spoiler: última)
* Podemos seguir el "método de Croston", que es un procedimiento basado en modelos en lugar de un procedimiento basado en datos. El método de Croston es vulnerable a valores atípicos y cambios / tendencias / cambios de nivel en la tasa de demanda, es decir, la demanda dividida por el número de períodos desde la última demanda.
* Un enfoque mucho más riguroso podría ser buscar "Datos dispersos - Datos desigualmente espaciados" o búsquedas como esa.
* Una solución bastante ingeniosa y simple es el smothing. Si una serie tiene puntos de tiempo en los que surgen ventas y largos períodos de tiempo en los que no surgen ventas, es posible convertir las ventas en ventas por período dividiendo las ventas observadas por el número de períodos sin ventas obteniendo así una tasa. Entonces es posible identificar un modelo entre la tasa y el intervalo entre las ventas que culminan en una tasa pronosticada y un intervalo pronosticado. Esto de manera más sencilla se transforma en una media movil.


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

Los resultados se pueden ver en results/stationary_test_manufacturing, acá un ejemplo de los 3 primeros, las series de tiempo trabajadas son estacionarias
```sh
Para la columna:  SP40
Rechaza la hipótesis nula (H0), los datos son estacionarios
ADF estadisticas: -6.757590
Valor de p: 0.000000
Valores criticos:
	1%: -3.436
	5%: -2.864
	10%: -2.568
Para la columna:  SP60-S
Rechaza la hipótesis nula (H0), los datos son estacionarios
ADF estadisticas: -5.848840
Valor de p: 0.000000
Valores criticos:
	1%: -3.436
	5%: -2.864
	10%: -2.568
Para la columna:  MP60-S
Rechaza la hipótesis nula (H0), los datos son estacionarios
ADF estadisticas: -5.881735
Valor de p: 0.000000
Valores criticos:
	1%: -3.436
	5%: -2.864
	10%: -2.568
```
# LSTM implementación

Los resultados de las redes lstm fueron bastante buenos en el dataset de moving_avarege, al mismo nivel que deep AR
De las 19 series de tiempo en 17/19 de las series este precento una mape menor 20 % y la accuracy promedio fue de **86.1 %** 
hay que seguir trabajando para bajar este error a través del preprocessing, dado que es el hecho que la demanda es muy intermintente
que hace que las predicciones sobre el conjunto de datos original no sean factibles.

**Resultados buenos**
![resultados de lstm](https://github.com/matheus695p/deep-ar/blob/master/results/lstm/rolling_DP30%20PIN_results.png)
![Screenshot](rolling_DP30%20PIN_results.png)


![resultados de lstm](https://github.com/matheus695p/deep-ar/blob/master/results/lstm/rolling_DP60%20PIN_results.png)
![resultados de lstm](https://github.com/matheus695p/deep-ar/blob/master/results/lstm/rolling_DQ60-S_results.png)
![resultados de lstm](https://github.com/matheus695p/deep-ar/blob/master/results/lstm/rolling_YPH48_results.png)

**Resultados malos**
![resultados de lstm](https://github.com/matheus695p/deep-ar/blob/master/results/lstm/rolling_DP30_results.png)

Estos resultados de lleno permiten hacer una mucho mejor planificación de la producción de como se hace actualmente.


#### deep AR resultados:

Después de varías pruebas, se determinó que la frecuencia de predicción sería de 14 días, es decir a partir de la última fecha en train, se predice 14 días al futuro, dos semanas, lo ideal es que se hagan predicciones cada 2 semanas, considerando los datos anteriores.

Acá se muestran un 






