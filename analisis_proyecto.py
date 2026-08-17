r"""
Proyecto final ESTG1034 - Influencia de los habitos cotidianos en el desempeno escolar
Dataset: "Student Habits vs Academic Performance: A Simulated Study" (Kaggle)
Variable respuesta: exam_score (cuantitativa continua, 0-100)

Este script usa unicamente tecnicas del alcance del curso:
  - medidas descriptivas y tablas de frecuencia
  - correlacion de Pearson y de Spearman
  - intervalo de confianza para la media
  - prueba t de diferencia de medias
  - prueba ji-cuadrado de independencia
  - regresion lineal simple y multiple (R2, R2 ajustado, error estandar, VIF)
  - diagnostico GRAFICO de residuos

Genera:
  salidas/resultados.txt  -> consola completa (para revisar e interpretar)
  salidas/figuras/*.pdf   -> graficos listos para \includegraphics
  salidas/tablas/*.tex    -> tablas listas para \input
"""

import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor

# =====================================================================
# 0. CONFIGURACION
# =====================================================================
ALPHA = 0.05

BASE = os.path.dirname(os.path.abspath(__file__))
DIR_SAL = os.path.join(BASE, "salidas")
DIR_FIG = os.path.join(DIR_SAL, "figuras")
DIR_TAB = os.path.join(DIR_SAL, "tablas")
for d in (DIR_SAL, DIR_FIG, DIR_TAB):
    os.makedirs(d, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 120,
    "savefig.bbox": "tight",
    "font.size": 9,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "axes.spines.top": False,
    "axes.spines.right": False,
})
COLOR = "#3A6EA5"
COLOR2 = "#C1666B"


class Tee:
    """Escribe en consola y en salidas/resultados.txt al mismo tiempo."""
    def __init__(self, path):
        self.file = open(path, "w", encoding="utf-8")
        self.stdout = sys.stdout

    def write(self, data):
        self.stdout.write(data)
        self.file.write(data)

    def flush(self):
        self.stdout.flush()
        self.file.flush()


sys.stdout = Tee(os.path.join(DIR_SAL, "resultados.txt"))


def titulo(txt):
    print("\n" + "=" * 78)
    print(txt)
    print("=" * 78)


def guardar_tabla(df, nombre, caption, label, index=True, decimales=3):
    """Exporta un DataFrame a .tex (booktabs, coma decimal) para \\input en Overleaf."""
    ruta = os.path.join(DIR_TAB, nombre + ".tex")
    d = df.copy()
    for col in d.columns:
        if pd.api.types.is_float_dtype(d[col]):
            d[col] = d[col].map(
                lambda v: "---" if pd.isna(v) else f"{v:.{decimales}f}".replace(".", ","))
        elif pd.api.types.is_integer_dtype(d[col]):
            d[col] = d[col].astype(str)
    for col in d.columns:
        d[col] = d[col].map(
            lambda v: str(v).replace("_", " ") if isinstance(v, str) and "$" not in v else v)
    ncols = len(d.columns) + (1 if index else 0)
    d.columns = [str(c).replace("%", "\\%").replace("_", " ") for c in d.columns]
    d.index = [str(i).replace("%", "\\%").replace("_", " ") for i in d.index]
    tex = d.to_latex(index=index, escape=False, column_format="l" + "r" * (ncols - 1),
                     caption=caption, label=label, position="H", longtable=False)
    tex = tex.replace("\\begin{table}[H]", "\\begin{table}[H]\n\\centering\n\\small")
    # Si la tabla excede el ancho de texto se reduce; si ya cabe, se deja igual.
    tex = tex.replace(
        "\\begin{tabular}",
        "\\resizebox{\\ifdim\\width>\\textwidth\\textwidth\\else\\width\\fi}{!}{%\n\\begin{tabular}")
    tex = tex.replace("\\end{tabular}", "\\end{tabular}}")
    with open(ruta, "w", encoding="utf-8") as fh:
        fh.write(tex)
    print("[tabla guardada] " + ruta)


def guardar_fig(fig, nombre):
    ruta = os.path.join(DIR_FIG, nombre + ".pdf")
    fig.savefig(ruta)
    plt.close(fig)
    print("[figura guardada] " + ruta)


# =====================================================================
# 1. CARGA DE DATOS
# =====================================================================
titulo("1. CARGA DE DATOS")

CSV_LOCAL = os.path.join(BASE, "student_habits_performance.csv")

if os.path.exists(CSV_LOCAL):
    df = pd.read_csv(CSV_LOCAL)
    print("Cargado desde archivo local:", CSV_LOCAL)
else:
    import kagglehub
    path = kagglehub.dataset_download("jayaantanaath/student-habits-vs-academic-performance")
    archivo = [f for f in os.listdir(path) if f.endswith(".csv")][0]
    df = pd.read_csv(os.path.join(path, archivo))
    print("Cargado desde kagglehub:", os.path.join(path, archivo))

print("Dimensiones (n, p):", df.shape)
print("\nPrimeras filas:")
print(df.head())

# =====================================================================
# 2. DEPURACION
# =====================================================================
titulo("2. DEPURACION DE DATOS")

print("Valores nulos por variable:")
nulos = df.isna().sum()
print(nulos[nulos > 0] if nulos.sum() > 0 else "Sin valores nulos")

print("\nFilas duplicadas completas:", df.duplicated().sum())
print("student_id duplicados:", df["student_id"].duplicated().sum())

# parental_education_level es la unica con faltantes: se conserva como
# categoria explicita para no perder observaciones.
if df["parental_education_level"].isna().sum() > 0:
    n_na = int(df["parental_education_level"].isna().sum())
    df["parental_education_level"] = df["parental_education_level"].fillna("No reportado")
    print(f"\n{n_na} faltantes en parental_education_level recodificados como 'No reportado'.")

orden_dieta = ["Poor", "Fair", "Good"]
orden_internet = ["Poor", "Average", "Good"]
df["diet_quality"] = pd.Categorical(df["diet_quality"], categories=orden_dieta, ordered=True)
df["internet_quality"] = pd.Categorical(df["internet_quality"], categories=orden_internet, ordered=True)

NUM = ["age", "study_hours_per_day", "social_media_hours", "netflix_hours",
       "attendance_percentage", "sleep_hours", "exercise_frequency",
       "mental_health_rating", "exam_score"]
CAT = ["gender", "part_time_job", "diet_quality", "parental_education_level",
       "internet_quality", "extracurricular_participation"]

print("\nRangos observados (min - max):")
for v in NUM:
    print(f"  {v:<28} {df[v].min():>8.2f}  -  {df[v].max():>8.2f}")

print("\nAtipicos segun criterio 1.5*RIC (diagrama de caja):")
for v in NUM:
    q1, q3 = df[v].quantile([0.25, 0.75])
    ric = q3 - q1
    fuera = ((df[v] < q1 - 1.5 * ric) | (df[v] > q3 + 1.5 * ric)).sum()
    print(f"  {v:<28} {fuera:>4} ({100*fuera/len(df):.1f} %)")

# La calificacion esta acotada en 100: conviene saber cuantos llegan al tope.
n_tope = int((df["exam_score"] >= 100).sum())
print(f"\nEstudiantes con la calificacion maxima de 100: {n_tope} ({100*n_tope/len(df):.1f} %)")
print("   La escala esta acotada por arriba; esto se retoma en el diagnostico del modelo.")

print("\nBase depurada: n =", len(df), " p =", df.shape[1])

# =====================================================================
# 3. ANALISIS DESCRIPTIVO
# =====================================================================
titulo("3. ANALISIS DESCRIPTIVO")

desc = pd.DataFrame({
    "n": df[NUM].count(),
    "Media": df[NUM].mean(),
    "Med.": df[NUM].median(),
    "D.E.": df[NUM].std(),
    "CV (%)": 100 * df[NUM].std() / df[NUM].mean(),
    "Min": df[NUM].min(),
    "Q1": df[NUM].quantile(0.25),
    "Q3": df[NUM].quantile(0.75),
    "Max": df[NUM].max(),
    "Asim.": df[NUM].skew(),
    "Curt.": df[NUM].kurtosis(),
})
print(desc.round(2).to_string())
guardar_tabla(desc.round(2), "desc_numericas",
              "Medidas descriptivas de las variables cuantitativas",
              "tab:desc_num", decimales=2)

print("\nDistribucion de frecuencias de las variables cualitativas:")
filas = []
for v in CAT:
    frec = df[v].value_counts(dropna=False)
    for cat, f in frec.items():
        filas.append({"Variable": v, "Categoria": str(cat),
                      "Frecuencia": int(f), "Porcentaje": round(100 * f / len(df), 1)})
tab_cat = pd.DataFrame(filas)
print(tab_cat.to_string(index=False))
guardar_tabla(tab_cat, "desc_categoricas",
              "Distribucion de frecuencias de las variables cualitativas",
              "tab:desc_cat", index=False, decimales=1)

# --- Figura 1: histogramas ---
clave = ["exam_score", "study_hours_per_day", "sleep_hours",
         "social_media_hours", "netflix_hours", "attendance_percentage"]
fig, axes = plt.subplots(2, 3, figsize=(9.5, 5.2))
for ax, v in zip(axes.ravel(), clave):
    ax.hist(df[v], bins=25, color=COLOR, edgecolor="white")
    ax.axvline(df[v].mean(), color=COLOR2, lw=1.4, ls="--",
               label=f"media = {df[v].mean():.2f}")
    ax.set_title(v.replace("_", " "), fontsize=9)
    ax.legend(fontsize=7, frameon=False)
fig.suptitle("Distribucion de las variables cuantitativas principales", fontsize=11)
fig.tight_layout()
guardar_fig(fig, "fig1_histogramas")

# --- Figura 2: cajas por grupos categoricos ---
fig, axes = plt.subplots(1, 3, figsize=(9.5, 3.2))
for ax, v in zip(axes, ["part_time_job", "diet_quality", "extracurricular_participation"]):
    cats = list(df[v].cat.categories) if str(df[v].dtype) == "category" else sorted(df[v].unique())
    datos = [df.loc[df[v] == c, "exam_score"].values for c in cats]
    bp = ax.boxplot(datos, patch_artist=True, widths=0.55)
    ax.set_xticks(range(1, len(cats) + 1))
    ax.set_xticklabels([str(c) for c in cats])
    for caja in bp["boxes"]:
        caja.set(facecolor=COLOR, alpha=0.55)
    for med in bp["medians"]:
        med.set(color=COLOR2, lw=1.6)
    ax.set_title(v.replace("_", " "), fontsize=9)
    ax.set_ylabel("exam_score" if v == "part_time_job" else "")
fig.suptitle("Calificacion final segun grupos categoricos", fontsize=11)
fig.tight_layout()
guardar_fig(fig, "fig2_cajas")

# =====================================================================
# 4. ANALISIS BIVARIANTE
# =====================================================================
titulo("4. ANALISIS BIVARIANTE")

corr = df[NUM].corr(method="pearson")
print("Matriz de correlacion de Pearson:")
print(corr.round(3).to_string())

filas = []
for v in NUM:
    if v == "exam_score":
        continue
    r, p = stats.pearsonr(df[v], df["exam_score"])
    rho, _ = stats.spearmanr(df[v], df["exam_score"])
    filas.append({"Variable": v, "r de Pearson": round(r, 3), "Valor p": p,
                  "rho de Spearman": round(rho, 3),
                  "Relacion": "significativa" if p < ALPHA else "no significativa"})
tab_corr = pd.DataFrame(filas).sort_values("r de Pearson", key=abs, ascending=False)
tab_corr["Valor p"] = tab_corr["Valor p"].apply(
    lambda x: "$<$ 0,001" if x < 0.001 else f"{x:.3f}".replace(".", ","))
print("\nCorrelacion de cada predictor con exam_score:")
print(tab_corr.to_string(index=False))
guardar_tabla(tab_corr, "correlaciones",
              "Correlacion de cada variable cuantitativa con la calificacion final",
              "tab:corr", index=False)

# --- Figura 3: mapa de calor ---
fig, ax = plt.subplots(figsize=(6.4, 5.4))
m = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(len(NUM)))
ax.set_xticklabels([v.replace("_", " ") for v in NUM], rotation=45, ha="right", fontsize=7)
ax.set_yticks(range(len(NUM)))
ax.set_yticklabels([v.replace("_", " ") for v in NUM], fontsize=7)
for i in range(len(NUM)):
    for j in range(len(NUM)):
        ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center",
                fontsize=6, color="black" if abs(corr.values[i, j]) < 0.6 else "white")
ax.grid(False)
fig.colorbar(m, ax=ax, shrink=0.8)
ax.set_title("Matriz de correlacion de Pearson", fontsize=10)
guardar_fig(fig, "fig3_correlacion")

# --- Figura 4: dispersiones ---
top = tab_corr["Variable"].head(3).tolist()
fig, axes = plt.subplots(1, 3, figsize=(9.5, 3.1))
for ax, v in zip(axes, top):
    ax.scatter(df[v], df["exam_score"], s=8, alpha=0.35, color=COLOR)
    b, a = np.polyfit(df[v], df["exam_score"], 1)
    xs = np.linspace(df[v].min(), df[v].max(), 50)
    ax.plot(xs, a + b * xs, color=COLOR2, lw=1.6)
    ax.set_xlabel(v.replace("_", " "))
    ax.set_title(f"r = {df[v].corr(df['exam_score']):.3f}", fontsize=9)
axes[0].set_ylabel("exam_score")
fig.suptitle("Relacion entre los habitos y la calificacion final", fontsize=11)
fig.tight_layout()
guardar_fig(fig, "fig4_dispersion")

# =====================================================================
# 5. INFERENCIA ESTADISTICA
# =====================================================================
titulo("5. INFERENCIA ESTADISTICA")

# --- 5.1 Intervalo de confianza para la media ---
x = df["exam_score"]
n = len(x)
ic = stats.t.interval(1 - ALPHA, df=n - 1, loc=x.mean(), scale=stats.sem(x))
print("5.1 Intervalo de confianza del 95 % para la media de exam_score")
print(f"    media = {x.mean():.3f} | s = {x.std(ddof=1):.3f} | n = {n}")
print(f"    IC 95 % = ({ic[0]:.3f} ; {ic[1]:.3f})")

# --- 5.2 Diferencia de medias segun part_time_job ---
print("\n5.2 Prueba t de diferencia de medias (exam_score ~ part_time_job)")
g1 = df.loc[df["part_time_job"] == "Yes", "exam_score"]
g0 = df.loc[df["part_time_job"] == "No", "exam_score"]
print(f"    Con trabajo (n={len(g1)}): media = {g1.mean():.3f}, s = {g1.std(ddof=1):.3f}")
print(f"    Sin trabajo (n={len(g0)}): media = {g0.mean():.3f}, s = {g0.std(ddof=1):.3f}")
print(f"    Diferencia de medias = {g1.mean() - g0.mean():.3f} puntos")
print(f"    Razon entre desviaciones = {max(g1.std(ddof=1), g0.std(ddof=1)) / min(g1.std(ddof=1), g0.std(ddof=1)):.3f}")
print("    (desviaciones muy similares: se justifica la prueba t con varianzas iguales)")
t_stat, t_p = stats.ttest_ind(g1, g0, equal_var=True)
print(f"    H0: mu_Yes = mu_No   vs   H1: mu_Yes != mu_No   (alpha = {ALPHA})")
print(f"    t = {t_stat:.3f}, gl = {len(g1) + len(g0) - 2}, p = {t_p:.4f}")
print("    Decision:", "se rechaza H0" if t_p < ALPHA else "no se rechaza H0")

# --- 5.3 Independencia (ji-cuadrado) ---
print("\n5.3 Prueba ji-cuadrado de independencia (diet_quality vs nivel de desempeno)")
df["nivel_desempeno"] = pd.qcut(df["exam_score"], 3, labels=["Bajo", "Medio", "Alto"])
tabla_cont = pd.crosstab(df["diet_quality"], df["nivel_desempeno"])
print(tabla_cont.to_string())
chi2, p_chi, gl_chi, esperadas = stats.chi2_contingency(tabla_cont)
print(f"    H0: las variables son independientes   (alpha = {ALPHA})")
print(f"    Chi2 = {chi2:.3f}, gl = {gl_chi}, p = {p_chi:.4f}")
print(f"    Frecuencia esperada minima = {esperadas.min():.2f} (debe ser >= 5)")
print("    Decision:", "se rechaza H0" if p_chi < ALPHA else "no se rechaza H0")
guardar_tabla(tabla_cont, "contingencia",
              "Tabla de contingencia: calidad de la dieta y nivel de desempeno",
              "tab:conting", decimales=0)

# =====================================================================
# 6. MODELO DE REGRESION LINEAL
# =====================================================================
titulo("6. MODELO DE REGRESION LINEAL")

print("6.1 Regresion lineal simple: exam_score ~ study_hours_per_day")
m_simple = smf.ols("exam_score ~ study_hours_per_day", data=df).fit()
print(m_simple.summary())

print("\n6.2 Regresion lineal multiple (modelo completo, las 14 predictoras)")
f_completo = (
    "exam_score ~ study_hours_per_day + social_media_hours + netflix_hours "
    "+ attendance_percentage + sleep_hours + exercise_frequency "
    "+ mental_health_rating + age + C(gender) + C(part_time_job) "
    "+ C(diet_quality) + C(internet_quality) + C(extracurricular_participation) "
    "+ C(parental_education_level)"
)
m_completo = smf.ols(f_completo, data=df).fit()
print(m_completo.summary())

print("\n6.3 Modelo reducido (solo los predictores significativos al 5 %)")
f_reducido = ("exam_score ~ study_hours_per_day + social_media_hours + netflix_hours "
              "+ attendance_percentage + sleep_hours + exercise_frequency "
              "+ mental_health_rating")
m_reducido = smf.ols(f_reducido, data=df).fit()
print(m_reducido.summary())

print("\n6.4 Comparacion de modelos (R2 ajustado y error estandar)")
comp = pd.DataFrame({
    "Modelo": ["Simple", "Multiple reducido", "Multiple completo"],
    "Variables": [1, 7, 14],
    "Terminos": [1, len(m_reducido.params) - 1, len(m_completo.params) - 1],
    "R2": [m_simple.rsquared, m_reducido.rsquared, m_completo.rsquared],
    "R2 ajustado": [m_simple.rsquared_adj, m_reducido.rsquared_adj, m_completo.rsquared_adj],
    "Error estandar": [np.sqrt(m_simple.mse_resid), np.sqrt(m_reducido.mse_resid),
                       np.sqrt(m_completo.mse_resid)],
    "F global": [m_simple.fvalue, m_reducido.fvalue, m_completo.fvalue],
})
print(comp.round(4).to_string(index=False))
print("\n   Variables = predictoras originales | Terminos = coeficientes estimados.")
print("   Difieren porque cada variable cualitativa se desagrega en variables")
print("   indicadoras: una menos que su numero de categorias.")
print("\nPrueba F global de cada modelo (H0: todos los coeficientes de pendiente son 0):")
for nom, mod in [("Simple", m_simple), ("Multiple reducido", m_reducido),
                 ("Multiple completo", m_completo)]:
    print(f"  {nom:<20} F = {mod.fvalue:>9.2f}  gl = ({int(mod.df_model)}, {int(mod.df_resid)})"
          f"  p = {mod.f_pvalue:.3e}  -> "
          f"{'se rechaza H0: el modelo es significativo' if mod.f_pvalue < ALPHA else 'no se rechaza H0'}")
guardar_tabla(comp.round(4), "comparacion_modelos",
              "Comparacion entre el modelo simple y los modelos multiples",
              "tab:modelos", index=False, decimales=4)

coefs = pd.DataFrame({
    "Coeficiente": m_reducido.params,
    "Error est.": m_reducido.bse,
    "t": m_reducido.tvalues,
    "Valor p": m_reducido.pvalues,
    "IC 95 % inf": m_reducido.conf_int()[0],
    "IC 95 % sup": m_reducido.conf_int()[1],
})
coefs.index = [i.replace("_", " ") for i in coefs.index]
coefs = coefs.round(4)
coefs["Valor p"] = coefs["Valor p"].apply(
    lambda v: "$<$ 0,001" if v < 0.001 else f"{v:.3f}".replace(".", ","))
print("\nCoeficientes del modelo reducido:")
print(coefs.to_string())
guardar_tabla(coefs, "coeficientes",
              "Coeficientes estimados del modelo de regresion lineal multiple",
              "tab:coef")

# Las variables medidas en HORAS si tienen coeficientes comparables entre si.
print("\nComparacion directa entre las variables medidas en horas diarias:")
for v in ["study_hours_per_day", "social_media_hours", "netflix_hours", "sleep_hours"]:
    print(f"  {v:<24} {m_reducido.params[v]:>8.3f} puntos por cada hora adicional")
print("   Las demas variables se miden en otras unidades, por lo que sus coeficientes")
print("   no son directamente comparables con estos.")

# =====================================================================
# 7. VALIDACION GRAFICA DEL MODELO
# =====================================================================
titulo("7. VALIDACION DEL MODELO")

modelo = m_reducido
resid = modelo.resid
ajust = modelo.fittedvalues
s_resid = np.sqrt(modelo.mse_resid)

print("Forma de los residuos (para contrastar con el grafico Q-Q):")
print(f"   Media = {resid.mean():.4f} (debe ser practicamente 0)")
print(f"   Desviacion estandar = {resid.std(ddof=1):.4f}")
print(f"   Asimetria = {stats.skew(resid):.3f}  |  Curtosis = {stats.kurtosis(resid):.3f}")
print("   (valores cercanos a 0 indican forma aproximadamente normal)")

print(f"\nResiduos extremos (mas alla de 3 errores estandar = {3*s_resid:.2f} puntos):")
extremos = int((np.abs(resid) > 3 * s_resid).sum())
print(f"   {extremos} observaciones ({100*extremos/len(df):.1f} %)")
print(f"   Residuo mas grande en valor absoluto: {np.abs(resid).max():.2f} puntos")

print("\nValores ajustados fuera de la escala valida 0-100:")
n_fuera = int((ajust > 100).sum())
print(f"   {n_fuera} observaciones con prediccion mayor a 100 ({100*n_fuera/len(df):.1f} %)")
print(f"   Valor ajustado maximo = {ajust.max():.2f}")
print("   Esto ocurre porque la calificacion real esta topada en 100.")

X = modelo.model.exog
vif = pd.DataFrame({
    "Variable": modelo.model.exog_names,
    "VIF": [variance_inflation_factor(X, i) for i in range(X.shape[1])],
})
vif = vif[vif["Variable"] != "Intercept"]
vif["Variable"] = vif["Variable"].str.replace("_", " ")
print("\nMulticolinealidad (VIF, se considera problematico si VIF > 5):")
print(vif.round(3).to_string(index=False))
guardar_tabla(vif.round(3), "vif",
              "Factores de inflacion de la varianza del modelo multiple",
              "tab:vif", index=False)

# --- Figura 5: diagnostico grafico ---
fig, axes = plt.subplots(1, 3, figsize=(9.5, 3.1))
axes[0].scatter(ajust, resid, s=8, alpha=0.35, color=COLOR)
axes[0].axhline(0, color=COLOR2, lw=1.3)
axes[0].set_xlabel("Valores ajustados")
axes[0].set_ylabel("Residuos")
axes[0].set_title("Residuos vs. ajustados", fontsize=9)

sm.qqplot(resid, line="45", fit=True, ax=axes[1], markerfacecolor=COLOR,
          markeredgecolor=COLOR, markersize=3, alpha=0.4)
axes[1].set_title("Grafico Q-Q de los residuos", fontsize=9)
axes[1].set_xlabel("Cuantiles teoricos")
axes[1].set_ylabel("Cuantiles muestrales")

axes[2].hist(resid, bins=30, color=COLOR, edgecolor="white", density=True)
xs = np.linspace(resid.min(), resid.max(), 200)
axes[2].plot(xs, stats.norm.pdf(xs, resid.mean(), resid.std()), color=COLOR2, lw=1.5)
axes[2].set_xlabel("Residuos")
axes[2].set_title("Distribucion de los residuos", fontsize=9)
fig.suptitle("Diagnostico grafico del modelo de regresion multiple", fontsize=11)
fig.tight_layout()
guardar_fig(fig, "fig5_diagnostico")

# =====================================================================
# 8. PREDICCION ILUSTRATIVA
# =====================================================================
titulo("8. PREDICCION ILUSTRATIVA")

perfiles = pd.DataFrame([
    {"nombre": "Estudiante promedio", "study_hours_per_day": 3.5, "social_media_hours": 2.5,
     "netflix_hours": 1.8, "attendance_percentage": 84.0, "sleep_hours": 6.5,
     "exercise_frequency": 3, "mental_health_rating": 5},
    {"nombre": "Habitos favorables", "study_hours_per_day": 4.5, "social_media_hours": 1.5,
     "netflix_hours": 1.0, "attendance_percentage": 90.0, "sleep_hours": 7.5,
     "exercise_frequency": 5, "mental_health_rating": 7},
    {"nombre": "Habitos desfavorables", "study_hours_per_day": 1.5, "social_media_hours": 5.0,
     "netflix_hours": 4.0, "attendance_percentage": 65.0, "sleep_hours": 4.5,
     "exercise_frequency": 0, "mental_health_rating": 2},
])

print("Verificacion de que los perfiles estan dentro del rango de los datos:")
for v in ["study_hours_per_day", "social_media_hours", "netflix_hours",
          "attendance_percentage", "sleep_hours", "exercise_frequency", "mental_health_rating"]:
    print(f"  {v:<24} rango observado: [{df[v].min():.1f}, {df[v].max():.1f}]"
          f" | perfiles: {perfiles[v].tolist()}")

pred = modelo.get_prediction(perfiles).summary_frame(alpha=ALPHA)
res_pred = pd.DataFrame({
    "Perfil": perfiles["nombre"],
    "Prediccion": pred["mean"].values,
    "IC inf": pred["mean_ci_lower"].values,
    "IC sup": pred["mean_ci_upper"].values,
})
print("\nPredicciones (calificacion sobre 100) con IC del 95 % para la media:")
print(res_pred.round(2).to_string(index=False))

print(f"\nDispersion de las predicciones individuales:")
print(f"   Error estandar residual del modelo = {s_resid:.2f} puntos.")
print("   Es la desviacion tipica de lo que un estudiante concreto se aparta")
print("   de la calificacion estimada por el modelo. El IC de la tabla mide")
print("   la precision de la MEDIA del grupo, no la de un estudiante individual.")

fuera = res_pred[(res_pred["Prediccion"] > 100) | (res_pred["Prediccion"] < 0)]
if len(fuera) > 0:
    print("\n[AVISO] Estos perfiles predicen fuera del rango valido 0-100:")
    print(fuera[["Perfil", "Prediccion"]].to_string(index=False))

guardar_tabla(res_pred.round(2), "prediccion",
              "Prediccion ilustrativa de la calificacion final para tres perfiles de habitos, "
              "con intervalo de confianza del 95\\% para la calificacion media del perfil.",
              "tab:pred", index=False, decimales=2)

sys.stdout.flush()