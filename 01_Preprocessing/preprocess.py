#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pré-processamento dos dados de diabetes para o algoritmo k-NN.

Estratégia:
1. Imputação dos valores faltantes (NaN) em Glucose, BloodPressure,
   SkinThickness, Insulin e BMI usando a mediana por classe (Outcome),
   já que essas colunas se comportam de forma diferente entre pacientes
   diabéticos e não diabéticos.
2. Normalização (padronização, z-score) de todas as features usando um
   StandardScaler ajustado apenas na base de treino (diabetes_dataset)
   e depois aplicado também na base de aplicação (diabetes_app), para
   manter as duas bases na mesma escala.

Sobrescreve diabetes_dataset.xlsx e diabetes_app.xlsx com as versões
tratadas.
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler

FEATURE_COLS = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
                'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
IMPUTE_COLS = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']

print(' - Lendo as bases originais')
data = pd.read_excel('diabetes_dataset.xlsx')
app = pd.read_excel('diabetes_app.xlsx')

print(' - Imputando valores faltantes (mediana por classe)')
for col in IMPUTE_COLS:
    data[col] = data.groupby('Outcome')[col].transform(lambda s: s.fillna(s.median()))

assert data[FEATURE_COLS].isna().sum().sum() == 0, 'ainda existem NaN após a imputação'

print(' - Normalizando as features (StandardScaler ajustado na base de treino)')
scaler = StandardScaler()
data[FEATURE_COLS] = scaler.fit_transform(data[FEATURE_COLS])
app[FEATURE_COLS] = scaler.transform(app[FEATURE_COLS])

print(' - Salvando os arquivos tratados')
data.to_excel('diabetes_dataset.xlsx', index=False)
app.to_excel('diabetes_app.xlsx', index=False)

print(' - Concluído')
