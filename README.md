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
│       caso_electricidad_DP30 PIN.png
│       caso_electricidad_DP30.png
│       caso_electricidad_DP60 PIN.png
│       caso_electricidad_DQ60-S.png
│       caso_electricidad_DQ60.png
│       caso_electricidad_IP41 LLAVE LAGRIMA.png
│       caso_electricidad_MP60-S.png
│       caso_electricidad_MP60.png
│       caso_electricidad_MT_001.png
│       caso_electricidad_MT_002.png
│       caso_electricidad_MT_003.png
│       caso_electricidad_MT_004.png
│       caso_electricidad_MT_005.png
│       caso_electricidad_MT_006.png
│       caso_electricidad_MT_007.png
│       caso_electricidad_MT_008.png
│       caso_electricidad_MT_009.png
│       caso_electricidad_MT_010.png
│       caso_electricidad_MT_011.png
│       caso_electricidad_MT_012.png
│       caso_electricidad_MT_013.png
│       caso_electricidad_MT_014.png
│       caso_electricidad_SP40 TU.png
│       caso_electricidad_SP40.png
│       caso_electricidad_SP60-S.png
│       caso_electricidad_SP60.png
│       caso_electricidad_YPH48.png
│       caso_electricidad_ZAPT44.png
│       caso_electricidad_ZAT48.png
│       caso_electricidad_ZIPP30.png
│       caso_electricidad_ZIPP44.png
│       caso_electricidad_ZP48.png
│       caso_manufactura_DP30 PIN.png
│       caso_manufactura_DP30.png
│       caso_manufactura_DP60 PIN.png
│       caso_manufactura_DQ60-S.png
│       caso_manufactura_DQ60.png
│       caso_manufactura_IP41 LLAVE LAGRIMA.png
│       caso_manufactura_MP60-S.png
│       caso_manufactura_MP60.png
│       caso_manufactura_SP40 TU.png
│       caso_manufactura_SP40.png
│       caso_manufactura_SP60-S.png
│       caso_manufactura_SP60.png
│       caso_manufactura_YPH48.png
│       caso_manufactura_ZAPT44.png
│       caso_manufactura_ZAT48.png
│       caso_manufactura_ZIPP30.png
│       caso_manufactura_ZIPP44.png
│       caso_manufactura_ZP48.png
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
