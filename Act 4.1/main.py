import requests
import numpy as np
import matplotlib.pyplot as plt
from fitter import Fitter
import scipy.stats as stats

# Fetch data from the API
url = "https://api.datos.gob.mx/v1/calidadAire"
params = {"pageSize": 500}
response = requests.get(url, params=params)
data = response.json()

# Process the data
valores = []
for resultado in data.get('results', []):
    estaciones = resultado.get('stations', [])
    for estacion in estaciones:
        for indice in estacion.get('indexes', []):
            if indice.get('scale') == 'IMECA':
                try:
                    valor = float(indice.get('value', 'NaN'))
                    if not np.isnan(valor):
                        valores.append(valor)
                except (ValueError, TypeError):
                    continue

valores_calidad_aire = np.array(valores)

# Create histogram
plt.figure(figsize=(10, 6))
plt.hist(valores_calidad_aire, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
plt.title('Histograma de Valores IMECA en Guadalajara')
plt.xlabel('Valor IMECA')
plt.ylabel('Frecuencia')
plt.grid(alpha=0.3)
plt.savefig('histograma_calidad_aire.png')
# plt.show()

f = Fitter(valores_calidad_aire)
f.fit()

# Get summary of results
resumen = f.summary(Nbest=5)
print("\nMejores distribuciones ajustadas:")
print(resumen)

# Plot histogram and best distributions
plt.figure(figsize=(12, 8))
f.hist()
f.plot_pdf(Nbest=3)
plt.title('Ajuste de Distribuciones para el Índice de Calidad del Aire')
plt.xlabel('Valor IMECA')
plt.ylabel('Densidad')
plt.grid(alpha=0.3)
plt.legend()
plt.savefig('ajuste_distribuciones.png')
# plt.show()

# Get parameters of the best distribution
mejor_dist = resumen.index[0]
mejores_params = f.fitted_param[mejor_dist]
print(f"\nMejor distribución: {mejor_dist}")
print(f"Parámetros: {mejores_params}")

# Basic statistics
print("\nEstadísticas básicas:")
print(f"Media: {np.mean(valores_calidad_aire):.2f}")
print(f"Mediana: {np.median(valores_calidad_aire):.2f}")
print(f"Desviación estándar: {np.std(valores_calidad_aire):.2f}")
print(f"Mínimo: {np.min(valores_calidad_aire):.2f}")
print(f"Máximo: {np.max(valores_calidad_aire):.2f}")
