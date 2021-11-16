# -*- coding: utf-8 -*-
"""
Created on Mon Nov 15 11:39:46 2021

@author: Jaime Mañas Galindo
Script para la creación de un modelo basado en "Random Forests" para predecir 
la calidad del aire en función de los parámetros recogidos por sensores.
A partir de un set de entrenamiento se crea el modelo y se evalúa.
El modelo recoge tres valores diferentes de etiqueta: 
    0, calidad del aire buena
    1, calidad del aire moderada
    2, calidad del aire peligrosa
Al introducir nuevos sets sin etiquetas se pueden realizar predicciones que 
devuelve en formato CSV.
"""
#Se importan las librerías necesarías
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

df_train = pd.read_csv('uoc_train.csv',sep=',') 
#Importación del set de entrenamiento, debe  encontrarse en la misma ubicación que el script.
#Separación del set en parámetros (X) y etiqueta (y)
X = df_train.values[:, :8] 
y = df_train.values[:, 8]
#parámetros y etiquetas son divididos aleatoriamente para obtener:
    #un set de entrenamiento con el que elaborar el modelo
    #un set de prueba con el contrastar resultados y poder evaluar
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    stratify=y, 
                                                    random_state=42)
#el parámetro 'random_state' permite fijar la división del set para reproducir 
#los resultados obtenidos

#Creación del modelo, se incluyen los parámeros:
    #'n_estimators': número de 'árboles' que presenta el modelo
    #'random_state': aleatoriedad fijada para reproducir resultados

forest = RandomForestClassifier(n_estimators=200, random_state=42)
forest.fit(X_train, y_train)
#Resultados predecidos para contrastar con los reales
y_test_pred = forest.predict(X_test)
#Devuelve una matriz con evaluaciones: precision, recall, f1-score, ...
print(classification_report(y_test, y_test_pred))
#Se ha obtenido el modelo que obtiene el mayor f1-score en base al parámetro 'n_estimators'

#Importación de set de parámetros sin etiqueta
X_pred = pd.read_csv('uoc_X_test.csv', sep=',')
y_pred = forest.predict(X_pred)
y_pred = np.array(y_pred, dtype='int')
df_pred = pd.DataFrame(y_pred)
#Se devuelve la predicción en formato CSV
df_pred.to_csv(r'pred.csv', index=False, header = ["output_prediction"])

