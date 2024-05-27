# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
import seaborn as sns
import numpy as np
import scipy as stats
import datetime as dt

fecha_A= dt.datetime.strptime('2024-03-31','%Y-%m-%d')
bases = []
df_mes = {}
rango = [1, 2, 3]

for mes in rango:
    base_posicion = f'/Users/Asus/OneDrive/Documentos/data trabajo/Rentabilidades_de_los_Fondos__FIC_{mes}.csv'
   
    bases.append(base_posicion)

for base, mes in zip(bases, rango):
    df_base = pd.read_csv(base)
    df_mes[mes] = df_base

   

base_concat = pd.concat(df_mes.values(), ignore_index=True)

base_concat.isnull().sum()

base_concat['FECHA_CORTE'] = pd.to_datetime(base_concat['FECHA_CORTE'],format= '%d/%m/%Y')
columns= ['NUMERO_UNIDADES_FONDO_CIERRE_OPER_DIA_T_ANTERIOR','NOMBRE_ENTIDAD', 'VALOR_UNIDAD_OPERACIONES_DIA_T','VALOR_FONDO_CIERRE_DIA_T', 'NUMERO_INVERSIONISTAS', 'RENTABILIDAD_DIARIA','RENTABILIDAD_MENSUAL', 'RENTABILIDAD_SEMESTRAL', 'RENTABILIDAD_ANUAL','RETIROS_REDENCIONES']
for col in columns:
    if col!= 'NOMBRE_ENTIDAD':
        base_concat[col] = base_concat[col].apply(lambda x: float(str(x).replace('$', '').replace(',', '')))
        base_concat = base_concat.loc[(base_concat['FECHA_CORTE']<= fecha_A)]
 

def remove_outliers_zscore(df, column):
    mean = df[column].mean()
    std = df[column].std()
    z_scores = np.abs((df[column] - mean) / std)
    threshold = 3  # Threshold to consider a value as an outlier
    return df[(z_scores < threshold)]

 

# Ejemplo de uso:

df_clean = remove_outliers_zscore(base_concat, 'RENTABILIDAD_MENSUAL')

df_Vcierre = base_concat.groupby([base_concat['FECHA_CORTE'].dt.to_period('M')])[['VALOR_FONDO_CIERRE_DIA_T', 'NUMERO_INVERSIONISTAS', 'RENTABILIDAD_MENSUAL']].mean()
df_Vopera =  base_concat.groupby([base_concat['FECHA_CORTE'].dt.to_period('M')])[['RENTABILIDAD_MENSUAL']].mean()


# Plot using Seaborn
sns.set(style='whitegrid')
plt.figure(figsize=(10, 6))
sns.lineplot(x='FECHA_CORTE', y='RENTABILIDAD_MENSUAL', data=base_concat, linestyle='--', marker='o')
plt.title('Variaciones Rentabilidad Promedio de FICs Colombia',fontsize=20)
plt.xlabel('Fecha',fontsize=15)
plt.ylabel('Variación Rentabilidad',fontsize=15)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()




entidades_mas_altas = base_concat.groupby('NOMBRE_ENTIDAD')['VALOR_UNIDAD_OPERACIONES_DIA_T'].max().nlargest(10)


entidades_extra = ['Valores Bancolombia S.A. Comisionista De Bolsa', 'Fiduciaria Bancolombia S.A. Sociedad Fiduciaria']

# Combinar las entidades de alto valor con las entidades adicionales
entidades_seleccionadas = entidades_mas_altas.index.tolist() + entidades_extra

# Filtrar el DataFrame original solo para las entidades seleccionadas
df_filtrado = base_concat[base_concat['NOMBRE_ENTIDAD'].isin(entidades_seleccionadas)]


# Graficar
plt.figure(figsize=(10,6))

# Iterar sobre los nombres de entidad únicos
for entidad in df_filtrado['NOMBRE_ENTIDAD'].unique():
    entidad_data = df_filtrado[df_filtrado['NOMBRE_ENTIDAD'] == entidad]
    plt.plot(entidad_data['FECHA_CORTE'], entidad_data['VALOR_UNIDAD_OPERACIONES_DIA_T'], label=entidad)

# Configurar leyenda, etiquetas de ejes, y título
plt.legend()
plt.xlabel('Fecha')
plt.ylabel('Valor de Unidad')
plt.title('Gráfico de Líneas por Fecha, Nombre de Entidad y Valor de Unidad')

# Rotar las fechas en el eje x para mayor legibilidad
plt.xticks(rotation=45)

# Mostrar el gráfico
plt.tight_layout()
plt.show()

tabla_pivot = pd.pivot_table(df_filtrado, values='VALOR_UNIDAD_OPERACIONES_DIA_T', index='NOMBRE_ENTIDAD', aggfunc='describe')


# Graficar
plt.figure(figsize=(10,6))

# Iterar sobre los nombres de entidad únicos
for entidad in df_filtrado['NOMBRE_ENTIDAD'].unique():
    entidad_data = df_filtrado[df_filtrado['NOMBRE_ENTIDAD'] == entidad]
    plt.plot(entidad_data['FECHA_CORTE'], entidad_data['VALOR_FONDO_CIERRE_DIA_T'], label=entidad)

# Configurar leyenda, etiquetas de ejes, y título
plt.legend()
plt.xlabel('Fecha')
plt.ylabel('Valor de cierre día t')
plt.title('Gráfico de Líneas por Fecha, Nombre de Entidad y Valor de Cierre')

# Rotar las fechas en el eje x para mayor legibilidad
plt.xticks(rotation=45)

# Mostrar el gráfico
plt.tight_layout()
plt.show()

tabla_pivot1 = pd.pivot_table(df_filtrado, values='VALOR_FONDO_CIERRE_DIA_T', index='NOMBRE_ENTIDAD', aggfunc='describe')


# Graficar
plt.figure(figsize=(10,6))

# Iterar sobre los nombres de entidad únicos
for entidad in df_filtrado['NOMBRE_ENTIDAD'].unique():
    entidad_data = df_filtrado[df_filtrado['NOMBRE_ENTIDAD'] == entidad]
    plt.plot(entidad_data['FECHA_CORTE'], entidad_data['NUMERO_INVERSIONISTAS'], label=entidad)

# Configurar leyenda, etiquetas de ejes, y título
plt.legend()
plt.xlabel('Fecha')
plt.ylabel('No inversionistas')
plt.title('Gráfico de Líneas por Fecha, Nombre de Entidad y No inversionistas')

# Rotar las fechas en el eje x para mayor legibilidad
plt.xticks(rotation=45)

# Mostrar el gráfico
plt.tight_layout()
plt.show()

tabla_pivot2 = pd.pivot_table(df_filtrado, values='NUMERO_INVERSIONISTAS', index='NOMBRE_ENTIDAD', aggfunc='describe')



plt.figure(figsize=(10,6))


# Iterar sobre los nombres de entidad únicos
for entidad in df_filtrado['NOMBRE_ENTIDAD'].unique():
    entidad_data = df_filtrado[df_filtrado['NOMBRE_ENTIDAD'] == entidad]
    plt.plot(entidad_data['FECHA_CORTE'], entidad_data['RENTABILIDAD_MENSUAL'], label=entidad)

# Configurar leyenda, etiquetas de ejes, y título
plt.legend()
plt.xlabel('Fecha')
plt.ylabel('Rentabilidad Mensual')
plt.title('Gráfico de Líneas por Fecha, Nombre de Entidad y Rentabilidad Mensual')

# Rotar las fechas en el eje x para mayor legibilidad
plt.xticks(rotation=45)

# Mostrar el gráfico
plt.tight_layout()
plt.show()
#%%
tabla_pivot3 = pd.pivot_table(df_filtrado, values='RENTABILIDAD_MENSUAL', index='NOMBRE_ENTIDAD', aggfunc='describe')

def mapea(v_guia, v_mapea, v_guia2):
    mapeo = dict(zip(v_guia,v_mapea))
    return v_guia2.map(mapeo)
#%%%
#Modelo
base_data  = pd.read_excel(r'\Users\Asus\OneDrive\Documentos\data trabajo\data3.xlsx')
base_data_daily = base_data[['Fecha', 'Tasa_interés', 'PIB','IPC','INVERSIÓN']]

base_fondos =pd.read_excel(r'\Users\Asus\OneDrive\Documentos\data trabajo\data2.xlsx')
columnas = [ 'Cód. SFC','Núm. Unidades','Valor unidad para las operaciones del día t','Valor fondo al cierre del día t', 'Núm. Invers.', 'Rentab. dia',
       'Rentab. mes', 'Rentab. sem', 'Rentab. año']
base_fondos['Valor fondo al cierre del día t'] = pd.to_numeric(base_fondos['Valor fondo al cierre del día t'], errors='coerce')

def procesar_columna(col):
    base_fondos[col] = pd.to_numeric(base_fondos[col], errors='coerce')
for col in columnas:
    procesar_columna(col)
    
perfiles_riesgo = base_fondos['Perfil de Riesgo'].unique()
dataframes_por_perfil = {}

for perfil in perfiles_riesgo:
    df_perfil = base_fondos[base_fondos['Perfil de Riesgo'] == perfil]
    dataframes_por_perfil[perfil] = df_perfil

Moderado = dataframes_por_perfil['Moderado']
Arriesgado = dataframes_por_perfil['Arriesgado']
Conservador = dataframes_por_perfil['Conservador']

df_grouped_Total = base_fondos.groupby(['Fecha'])[['Núm. Unidades','Valor unidad para las operaciones del día t','Valor fondo al cierre del día t', 'Núm. Invers.', 'Rentab. dia',
       'Rentab. dia', 'Rentab. sem', 'Rentab. año']].sum()
df_grouped_M = Moderado.groupby(['Fecha', 'Perfil de Riesgo'])[['Núm. Unidades','Valor unidad para las operaciones del día t','Valor fondo al cierre del día t', 'Núm. Invers.', 'Rentab. dia',
       'Rentab. mes', 'Rentab. sem', 'Rentab. año']].sum()
df_grouped_A = Arriesgado.groupby(['Fecha', 'Perfil de Riesgo'])[['Núm. Unidades','Valor unidad para las operaciones del día t','Valor fondo al cierre del día t', 'Núm. Invers.', 'Rentab. dia',
       'Rentab. mes', 'Rentab. sem', 'Rentab. año']].sum()
df_grouped_C = Conservador.groupby(['Fecha', 'Perfil de Riesgo'])[['Núm. Unidades','Valor unidad para las operaciones del día t','Valor fondo al cierre del día t', 'Núm. Invers.', 'Rentab. dia',
       'Rentab. mes', 'Rentab. sem', 'Rentab. año']].sum()

df_total =pd.merge(df_grouped_Total, base_data_daily, on='Fecha')

df_moderado =pd.merge(df_grouped_M, base_data_daily, on='Fecha')
df_moderado['Perfil de Riesgo'] =mapea(Moderado['Fecha'],Moderado['Perfil de Riesgo'],df_moderado['Fecha'])

df_arriesgado =pd.merge(df_grouped_A, base_data_daily, on='Fecha')
df_arriesgado['Perfil de Riesgo'] =mapea(Arriesgado['Fecha'],Arriesgado['Perfil de Riesgo'],df_arriesgado['Fecha'])

df_conservador =pd.merge(df_grouped_C, base_data_daily, on='Fecha')
df_conservador['Perfil de Riesgo'] =mapea(Conservador['Fecha'],Conservador['Perfil de Riesgo'],df_conservador['Fecha'])


#%%
import pandas as pd

df_total = df_total[['Fecha', 'IPC','Valor fondo al cierre del día t','Tasa_interés', 'PIB','INVERSIÓN']]
# Assuming df_total is your DataFrame containing the data
df_total['INVERSIÓN'] = df_total['INVERSIÓN'] * df_total['PIB'] / 100

# Define the function to simulate the impact of FIC on aggregate investment
def simulate_investment(ipc, fic_value, tasa_interes, pib, inversion):
    # Simulate the impact of FIC on aggregate investment
    fic_value_billions = fic_value / 1000000
   
   # Convert IPC to percentage
    ipc_percentage = ipc / 100
    investment = pib * (1 + tasa_interes) * (1 - ipc_percentage) * (1 + fic_value_billions / pib)
    return investment

# Define the number of scenarios
num_scenarios = 1000

# Create an array to store the scenarios
scenarios = np.zeros((num_scenarios, 5))

# Generate the scenarios using random sampling
for i in range(num_scenarios):
    ipc = np.random.normal(df_total['IPC'].mean(), df_total['IPC'].std())
    fic_value = np.random.normal(df_total['Valor fondo al cierre del día t'].mean(), df_total['Valor fondo al cierre del día t'].std())
    tasa_interes = np.random.normal(df_total['Tasa_interés'].mean(), df_total['Tasa_interés'].std())
    pib = np.random.normal(df_total['PIB'].mean(), df_total['PIB'].std())
    inversion = np.random.normal(df_total['INVERSIÓN'].mean(), df_total['INVERSIÓN'].std())
    scenarios[i] = [ipc, fic_value, tasa_interes, pib, inversion]

# Simulate the impact of FIC on aggregate investment for each scenario
investments = np.array([simulate_investment(*scenario) for scenario in scenarios])

# Calculate the mean and standard deviation of the investments
mean_investment = np.mean(investments)
std_investment = np.std(investments)

print(f'Mean investment: {mean_investment:.2f}')
print(f'Standard deviation of investment: {std_investment:.2f}')

import matplotlib.pyplot as plt

plt.hist(investments, bins=50, alpha=0.5, label='Distribución de la inversión')
plt.xlabel('Inversión (millones de pesos)')
plt.ylabel('Frecuencia')
plt.title('Distribución de la inversión')
plt.show()



plt.boxplot(investments, vert=True, patch_artist=True)
plt.xlabel('Inversión (millones de pesos)')
plt.title('Gráfico de caja y bigotes de la inversión')
plt.show()



plt.plot(investments)
plt.xlabel('Escenario')
plt.ylabel('Inversión (millones de pesos)')
plt.title('Evolución de la inversión en cada escenario')
plt.show()


plt.bar(['Media', 'Desviación estándar'], [mean_investment, std_investment])
plt.xlabel('Estadística')
plt.ylabel('Inversión (millones de pesos)')
plt.title('Inversión media y desviación estándar')
plt.show()



plt.scatter(scenarios[:, 3], investments)
plt.xlabel('PIB (billones de pesos)')
plt.ylabel('Inversión (millones de pesos)')
plt.title('Relación entre la inversión y el PIB')
plt.show()
#%%moderado

df_moderado = df_moderado[['Fecha', 'IPC','Valor fondo al cierre del día t','Tasa_interés', 'PIB','INVERSIÓN']]
# Assuming df_total is your DataFrame containing the data
df_moderado['INVERSIÓN'] = df_moderado['INVERSIÓN'] * df_moderado['PIB'] / 100

# Define the function to simulate the impact of FIC on aggregate investment
def simulate_investment(ipc, fic_value, tasa_interes, pib, inversion):
    # Simulate the impact of FIC on aggregate investment
    fic_value_billions = fic_value / 1000000
   
   # Convert IPC to percentage
    ipc_percentage = ipc / 100
    investment = pib * (1 + tasa_interes) * (1 - ipc_percentage) * (1 + fic_value_billions / pib)
    return investment

# Define the number of scenarios
num_scenarios = 1000

# Create an array to store the scenarios
scenarios = np.zeros((num_scenarios, 5))

# Generate the scenarios using random sampling
for i in range(num_scenarios):
    ipc = np.random.normal(df_moderado['IPC'].mean(), df_moderado['IPC'].std())
    fic_value = np.random.normal(df_moderado['Valor fondo al cierre del día t'].mean(), df_moderado['Valor fondo al cierre del día t'].std())
    tasa_interes = np.random.normal(df_moderado['Tasa_interés'].mean(), df_moderado['Tasa_interés'].std())
    pib = np.random.normal(df_moderado['PIB'].mean(), df_moderado['PIB'].std())
    inversion = np.random.normal(df_moderado['INVERSIÓN'].mean(), df_moderado['INVERSIÓN'].std())
    scenarios[i] = [ipc, fic_value, tasa_interes, pib, inversion]

# Simulate the impact of FIC on aggregate investment for each scenario
investments = np.array([simulate_investment(*scenario) for scenario in scenarios])

# Calculate the mean and standard deviation of the investments
mean_investment = np.mean(investments)
std_investment = np.std(investments)

print(f'Mean investment: {mean_investment:.2f}')
print(f'Standard deviation of investment: {std_investment:.2f}')
import matplotlib.pyplot as plt

plt.hist(investments, bins=50, alpha=0.5, label='Distribución de la inversión')
plt.xlabel('Inversión (millones de pesos)')
plt.ylabel('Frecuencia')
plt.title('Distribución de la inversión')
plt.show()



plt.boxplot(investments, vert=True, patch_artist=True)
plt.xlabel('Inversión (millones de pesos)')
plt.title('Gráfico de caja y bigotes de la inversión')
plt.show()



plt.plot(investments)
plt.xlabel('Escenario')
plt.ylabel('Inversión (millones de pesos)')
plt.title('Evolución de la inversión en cada escenario')
plt.show()

#%% arriesgado

df_arriesgado = df_arriesgado[['Fecha', 'IPC','Valor fondo al cierre del día t','Tasa_interés', 'PIB','INVERSIÓN']]
# Assuming df_total is your DataFrame containing the data
df_arriesgado['INVERSIÓN'] = df_arriesgado['INVERSIÓN'] * df_arriesgado['PIB'] / 100

# Define the function to simulate the impact of FIC on aggregate investment
def simulate_investment(ipc, fic_value, tasa_interes, pib, inversion):
    # Simulate the impact of FIC on aggregate investment
    fic_value_billions = fic_value / 1000000
   
   # Convert IPC to percentage
    ipc_percentage = ipc / 100
    investment = pib * (1 + tasa_interes) * (1 - ipc_percentage) * (1 + fic_value_billions / pib)
    return investment

# Define the number of scenarios
num_scenarios = 1000

# Create an array to store the scenarios
scenarios = np.zeros((num_scenarios, 5))

# Generate the scenarios using random sampling
for i in range(num_scenarios):
    ipc = np.random.normal(df_arriesgado['IPC'].mean(), df_arriesgado['IPC'].std())
    fic_value = np.random.normal(df_arriesgado['Valor fondo al cierre del día t'].mean(), df_arriesgado['Valor fondo al cierre del día t'].std())
    tasa_interes = np.random.normal(df_arriesgado['Tasa_interés'].mean(), df_arriesgado['Tasa_interés'].std())
    pib = np.random.normal(df_arriesgado['PIB'].mean(), df_arriesgado['PIB'].std())
    inversion = np.random.normal(df_arriesgado['INVERSIÓN'].mean(), df_arriesgado['INVERSIÓN'].std())
    scenarios[i] = [ipc, fic_value, tasa_interes, pib, inversion]

# Simulate the impact of FIC on aggregate investment for each scenario
investments = np.array([simulate_investment(*scenario) for scenario in scenarios])

# Calculate the mean and standard deviation of the investments
mean_investment = np.mean(investments)
std_investment = np.std(investments)

print(f'Mean investment: {mean_investment:.2f}')
print(f'Standard deviation of investment: {std_investment:.2f}')

import matplotlib.pyplot as plt

plt.hist(investments, bins=50, alpha=0.5, label='Distribución de la inversión')
plt.xlabel('Inversión (millones de pesos)')
plt.ylabel('Frecuencia')
plt.title('Distribución de la inversión')
plt.show()



plt.boxplot(investments, vert=True, patch_artist=True)
plt.xlabel('Inversión (millones de pesos)')
plt.title('Gráfico de caja y bigotes de la inversión')
plt.show()



plt.plot(investments)
plt.xlabel('Escenario')
plt.ylabel('Inversión (millones de pesos)')
plt.title('Evolución de la inversión en cada escenario')
plt.show()


#%% conservador

df_conservador = df_conservador[['Fecha', 'IPC','Valor fondo al cierre del día t','Tasa_interés', 'PIB','INVERSIÓN']]
# Assuming df_total is your DataFrame containing the data
df_conservador['INVERSIÓN'] = df_conservador['INVERSIÓN'] * df_conservador['PIB'] / 100

# Define the function to simulate the impact of FIC on aggregate investment
def simulate_investment(ipc, fic_value, tasa_interes, pib, inversion):
    # Simulate the impact of FIC on aggregate investment
    fic_value_billions = fic_value / 1000000
   
   # Convert IPC to percentage
    ipc_percentage = ipc / 100
    investment = pib * (1 + tasa_interes) * (1 - ipc_percentage) * (1 + fic_value_billions / pib)
    return investment

# Define the number of scenarios
num_scenarios = 1000

# Create an array to store the scenarios
scenarios = np.zeros((num_scenarios, 5))

# Generate the scenarios using random sampling
for i in range(num_scenarios):
    ipc = np.random.normal(df_conservador['IPC'].mean(), df_conservador['IPC'].std())
    fic_value = np.random.normal(df_conservador['Valor fondo al cierre del día t'].mean(), df_conservador['Valor fondo al cierre del día t'].std())
    tasa_interes = np.random.normal(df_conservador['Tasa_interés'].mean(), df_conservador['Tasa_interés'].std())
    pib = np.random.normal(df_conservador['PIB'].mean(), df_conservador['PIB'].std())
    inversion = np.random.normal(df_conservador['INVERSIÓN'].mean(), df_conservador['INVERSIÓN'].std())
    scenarios[i] = [ipc, fic_value, tasa_interes, pib, inversion]

# Simulate the impact of FIC on aggregate investment for each scenario
investments = np.array([simulate_investment(*scenario) for scenario in scenarios])

# Calculate the mean and standard deviation of the investments
mean_investment = np.mean(investments)
std_investment = np.std(investments)

print(f'Mean investment: {mean_investment:.2f}')
print(f'Standard deviation of investment: {std_investment:.2f}')

import matplotlib.pyplot as plt

plt.hist(investments, bins=50, alpha=0.5, label='Distribución de la inversión')
plt.xlabel('Inversión (millones de pesos)')
plt.ylabel('Frecuencia')
plt.title('Distribución de la inversión')
plt.show()



plt.boxplot(investments, vert=True, patch_artist=True)
plt.xlabel('Inversión (millones de pesos)')
plt.title('Gráfico de caja y bigotes de la inversión')
plt.show()



plt.plot(investments)
plt.xlabel('Escenario')
plt.ylabel('Inversión (millones de pesos)')
plt.title('Evolución de la inversión en cada escenario')
plt.show()

df_total =pd.merge(df_grouped_Total, base_data_daily, on='Fecha')
# Datos de ejemplo
df_rateI = df_total[['Fecha','Núm. Invers.', 'Rentab. dia','Tasa_interés']]

plt.plot(df_rateI['Fecha'], df_rateI['Rentab. dia'], label='Rentabilidad')
plt.plot(df_rateI['Fecha'], df_rateI['Tasa_interés'], label='Tasa de Interés')

# Agregar título y etiquetas a los ejes
plt.title('Rentabilidad diaria y Tasa de Interés en el Tiempo')
plt.xlabel('Año')
plt.ylabel('Valor')

# Agregar leyenda
plt.legend()

# Mostrar el gráfico
plt.show()

# Crear un gráfico de líneas para la rentabilidad
plt.figure(figsize=(10, 5))
plt.plot(df_rateI['Fecha'],df_rateI['Valor unidad para las operaciones del día t'], label='Valor Unidad')
plt.title('Valor Unidad en el Tiempo')
plt.xlabel('Año')
plt.ylabel('Valor')
plt.legend()
plt.show()

# Crear un gráfico de líneas para la tasa de interés
plt.figure(figsize=(10, 5))
plt.plot(df_rateI['Fecha'],df_rateI['Tasa_interés'], label='Tasa de Interés')
plt.title('Tasa de Interés en el Tiempo')
plt.xlabel('Año')
plt.ylabel('Valor')
plt.legend()
plt.show()

