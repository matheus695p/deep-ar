[![Build Status](https://travis-ci.org/joemccann/dillinger.svg?branch=master)](https://travis-ci.org/joemccann/dillinger)

# deep AR

Paper deep AR de amazon, implementación a través de gluonts y MX-NET



### Instalar las librerías necesarias para hacer el testing
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
