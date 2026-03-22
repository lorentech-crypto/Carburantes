"""
Actividad 1: Exploración y preprocesamiento de datos con un dataset de precios de gasolina por provincias españolas (2020-2023)

"""

# 0. IMPORTACIONES

from xml.etree.ElementTree import VERSION
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
import os, warnings
warnings.filterwarnings('ignore')


# 1. GENERACIÓN / CARGA DEL DATASET

np.random.seed(42)

provincias = [
    'Madrid', 'Barcelona', 'Valencia', 'Sevilla', 'Zaragoza',
    'Bilbao', 'Malaga', 'Murcia', 'Alicante', 'Valladolid',
    'Palma', 'Las Palmas', 'Santa Cruz de Tenerife', 'Cordoba', 'Granada'
]
fechas = pd.date_range('2020-01-01', '2023-12-31', freq='MS')
base_95, base_98, base_d, base_dp = 1.25, 1.38, 1.15, 1.28

rows = []
for fecha in fechas:
    t = (fecha.year - 2020) * 12 + fecha.month
    tendencia      = 0.004 * t
    crisis_energy  = 0.18 if fecha.year == 2022 else 0.0
    estacionalidad = 0.03 * np.sin(2 * np.pi * t / 12)
    for prov in provincias:
        fp = np.random.uniform(-0.05, 0.05)
        region = ('Norte' if prov in ['Bilbao','Zaragoza','Valladolid'] else
                  'Sur'   if prov in ['Sevilla','Malaga','Cordoba','Granada'] else
                  'Islas' if prov in ['Palma','Las Palmas','Santa Cruz de Tenerife']
                  else 'Centro-Levante')
        rows.append({
            'fecha'             : fecha,
            'provincia'         : prov,
            'region'            : region,
            'precio_95'         : round(base_95 + tendencia + crisis_energy + estacionalidad + fp + np.random.normal(0,0.015),3),
            'precio_98'         : round(base_98 + tendencia + crisis_energy + estacionalidad + fp + np.random.normal(0,0.015),3),
            'precio_diesel'     : round(base_d  + tendencia + crisis_energy + estacionalidad + fp + np.random.normal(0,0.012),3),
            'precio_diesel_plus': round(base_dp + tendencia + crisis_energy + estacionalidad + fp + np.random.normal(0,0.012),3),
            'num_estaciones'    : int(np.random.normal(180,60)),
            'impuesto_especial' : round(np.random.uniform(0.40,0.48),3),
        })

df = pd.DataFrame(rows)

# Introducir ~7% valores faltantes
for col in ['precio_95','precio_98','precio_diesel','num_estaciones']:
    mask = np.random.rand(len(df)) < 0.07
    df.loc[mask, col] = np.nan

df.to_csv('gasolina_espana.csv', index=False)
print(f"Dataset generado: {df.shape[0]} filas x {df.shape[1]} columnas")


# 2. EXPLORACIÓN INICIAL

print("\n--- INFORMACIÓN DEL DATASET ---")
df.info()
print("\n--- ESTADÍSTICAS DESCRIPTIVAS ---")
print(df.describe().round(3))
print("\n--- VALORES NULOS ---")
print(df.isnull().sum())
print(f"\nRango temporal: {df.fecha.min().date()} → {df.fecha.max().date()}")
print("Provincias:", sorted(df.provincia.unique()))

# 3. FILTRADO POR PROVINCIA Y RANGO DE PRECIO

df_madrid = df[df['provincia'] == 'Madrid'].copy()
print(f"\nRegistros Madrid: {len(df_madrid)}")

df_2022 = df[(df['fecha'] >= '2022-01-01') & (df['fecha'] <= '2022-12-31')]
print(f"Registros 2022: {len(df_2022)}")

precio_min, precio_max = 1.30, 1.55
df_rango = df[(df['precio_95'] >= precio_min) & (df['precio_95'] <= precio_max)]
print(f"precio_95 entre {precio_min}-{precio_max} EUR: {len(df_rango)} registros")

df_baratos = df[df['precio_95'] < 1.30]
df_caros   = df[df['precio_95'] > 1.55]
print(f"precio_95 < 1.30: {len(df_baratos)}  |  precio_95 > 1.55: {len(df_caros)}")


# 4. VISUALIZACIÓN CON MATPLOTLIB

os.makedirs('plots', exist_ok=True)

# 4.1 Serie temporal Madrid
fig, ax = plt.subplots(figsize=(11,4))
ax.plot(df_madrid['fecha'], df_madrid['precio_95'],  color='#2E7D32', lw=2, label='Gasolina 95')
ax.plot(df_madrid['fecha'], df_madrid['precio_diesel'], color='#1565C0', lw=2, linestyle='--', label='Diesel')
ax.axvspan(pd.Timestamp('2022-01-01'), pd.Timestamp('2022-12-31'), alpha=0.12, color='red', label='Crisis 2022')
ax.set_title('Evolucion del precio de combustible — Madrid', fontweight='bold')
ax.set_ylabel('EUR/litro'); ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout(); plt.savefig('plots/01_serie_temporal_madrid.png', dpi=150, bbox_inches='tight')
plt.show(); print("Grafica 1 guardada.")

# 4.2 Histogramas
fig, axes = plt.subplots(1,4, figsize=(14,4))
for ax, col, lbl, c in zip(axes,
    ['precio_95','precio_98','precio_diesel','precio_diesel_plus'],
    ['Gasolina 95','Gasolina 98','Diesel','Diesel+'],
    ['#2E7D32','#1565C0','#E65100','#C62828']):
    data = df[col].dropna()
    ax.hist(data, bins=28, color=c, edgecolor='white', alpha=0.85)
    ax.axvline(data.mean(),   color='black', lw=1.5, linestyle='--', label=f'Media: {data.mean():.3f}')
    ax.axvline(data.median(), color='gray',  lw=1.5, linestyle=':',  label=f'Mediana: {data.median():.3f}')
    ax.set_title(lbl, fontsize=10, fontweight='bold'); ax.set_xlabel('EUR/litro'); ax.legend(fontsize=8)
fig.suptitle('Distribucion de precios por tipo de combustible', fontweight='bold')
plt.tight_layout(); plt.savefig('plots/02_histogramas.png', dpi=150, bbox_inches='tight')
plt.show(); print("Grafica 2 guardada.")

# 4.3 Boxplot por región
fig, ax = plt.subplots(figsize=(9,5))
df.boxplot(column='precio_95', by='region', ax=ax, patch_artist=True,
           boxprops=dict(facecolor='#A5D6A7'), medianprops=dict(color='#2E7D32', lw=2))
ax.set_title('Precio Gasolina 95 por Region', fontweight='bold')
ax.set_xlabel('Region'); ax.set_ylabel('EUR/litro'); plt.suptitle('')
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout(); plt.savefig('plots/03_boxplot_region.png', dpi=150, bbox_inches='tight')
plt.show(); print("Grafica 3 guardada.")

# 4.4 Matriz de correlación
num_cols = ['precio_95','precio_98','precio_diesel','precio_diesel_plus','num_estaciones','impuesto_especial']
corr = df[num_cols].corr()
fig, ax = plt.subplots(figsize=(8,6))
sns.heatmap(corr, mask=np.triu(np.ones_like(corr,dtype=bool)), annot=True, fmt='.2f',
            cmap='RdYlGn', center=0, square=True, ax=ax, linewidths=0.5)
ax.set_title('Matriz de Correlacion', fontweight='bold')
plt.tight_layout(); plt.savefig('plots/04_correlacion.png', dpi=150, bbox_inches='tight')
plt.show(); print("Grafica 4 guardada.")

# 4.5 Precio medio por provincia
media_prov = df.groupby('provincia')['precio_95'].mean().sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(12,4))
bars = ax.bar(media_prov.index, media_prov.values, color='#2E7D32', edgecolor='white', alpha=0.85)
ax.bar_label(bars, fmt='%.3f', fontsize=8, padding=2)
ax.set_title('Precio Medio Gasolina 95 por Provincia', fontweight='bold')
ax.set_ylabel('EUR/litro'); ax.grid(True, alpha=0.3, axis='y')
plt.xticks(rotation=30, ha='right')
plt.tight_layout(); plt.savefig('plots/05_media_provincia.png', dpi=150, bbox_inches='tight')
plt.show(); print("Grafica 5 guardada.")

# 5. PREPROCESAMIENTO

# 5.1 Imputación con la media
print("\n--- IMPUTACIÓN (SimpleImputer, strategy='mean') ---")
print("Nulos ANTES:\n", df.isnull().sum())
cols_impute = ['precio_95','precio_98','precio_diesel','num_estaciones']
imputer = SimpleImputer(strategy='mean')
df[cols_impute] = imputer.fit_transform(df[cols_impute])
print("Nulos DESPUÉS:\n", df.isnull().sum())

# 5.2 Corrección de anomalías (num_estaciones negativo)
anomalos = (df['num_estaciones'] < 0).sum()
print(f"\nValores negativos en num_estaciones: {anomalos}")
df['num_estaciones'] = df['num_estaciones'].clip(lower=0)

# 5.3 Normalización MinMax
print("\n--- NORMALIZACIÓN MinMax ---")
scaler = MinMaxScaler()
cols_norm = ['precio_95','precio_98','precio_diesel','precio_diesel_plus','impuesto_especial']
df_norm = df.copy()
df_norm[cols_norm] = scaler.fit_transform(df[cols_norm])
print(df_norm[cols_norm].describe().loc[['min','max']].round(4))

# Visualizar normalización
fig, axes = plt.subplots(1,2, figsize=(12,4))
df[['precio_95','precio_diesel']].plot.hist(bins=25, ax=axes[0], alpha=0.7, color=['#2E7D32','#1565C0'])
axes[0].set_title('Antes de Normalizar', fontweight='bold'); axes[0].set_xlabel('EUR/litro')
df_norm[['precio_95','precio_diesel']].plot.hist(bins=25, ax=axes[1], alpha=0.7, color=['#2E7D32','#1565C0'])
axes[1].set_title('Despues de Normalizar (MinMax)', fontweight='bold'); axes[1].set_xlabel('Valor [0-1]')
plt.suptitle('Efecto de la Normalizacion MinMax', fontweight='bold')
plt.tight_layout(); plt.savefig('plots/06_normalizacion.png', dpi=150, bbox_inches='tight')
plt.show(); print("Grafica 6 guardada.")


# 6. EDA AUTOMÁTICO CON AUTOVIZ

try:
    from autoviz.AutoViz_Class import AutoViz_Class
    AV = AutoViz_Class()
    dft = AV.AutoViz(
        filename='', sep=',', depVar='precio_95',
        dfte=df.drop(columns=['fecha','provincia','region'], errors='ignore'),
        header=0, verbose=1, lowess=False,
        chart_format='png', max_rows_analyzed=720, max_cols_analyzed=20,
    )
    print("EDA AutoViz completado. Graficas en AutoViz_Plots/")
except ImportError:
    print("AVISO: AutoViz no instalado. Ejecuta: pip install autoviz")


# 7. CORRELACIONES

print("\n--- CORRELACIONES CON precio_95 ---")
print(df[num_cols].corr()['precio_95'].sort_values(ascending=False).round(3))
print("Completado con exito!")
