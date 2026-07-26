"""
NOAA Storm Events — Consolidación, limpieza, EDA y construcción de dataset maestro.

Convertido desde el notebook `1_storm_events_final.ipynb` a un script con funciones,
siguiendo el patrón: cada celda del notebook pasa a ser una función con nombre
descriptivo, y `main()` las encadena en orden.

Notas sobre la conversión:
- Se removió el bloque de Google Colab (`from google.colab import drive; drive.mount(...)`)
  porque es específico de ese entorno y no tiene sentido dentro de un contenedor de
  Airflow. `DATA_DIRS` y `OUTPUT_DIR` quedan como rutas locales relativas — vos
  decidís si las adaptás a rutas de un bucket de MinIO (como charlamos, usando
  `s3fs`/`boto3`) o si las dejás como bind mount local.
- Las funciones de EDA devuelven la figura de matplotlib (`fig`) en vez de llamar
  a `plt.show()`, para que puedas elegir cómo mostrarlas: en un notebook (`fig`
  se muestra solo), guardadas a disco (`fig.savefig(...)`), o logueadas a MLflow
  (`mlflow.log_figure(fig, "nombre.png")`, como vimos antes).
- Todo el resto de la lógica (parseo de daños, fechas, features, target) se
  mantiene igual al notebook original — no se cambió ningún criterio metodológico.
"""

import os
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import s3fs


# ---------------------------------------------------------------------------
# 1. Configuración
# ---------------------------------------------------------------------------

# Estas variables ya están disponibles en el contenedor de Airflow (se las
# pasamos en el docker-compose.yml). Si corrés el script suelto fuera de
# Airflow, exportalas antes en tu terminal.
MINIO_ENDPOINT = os.environ["MLFLOW_S3_ENDPOINT_URL"]   # ej: http://minio:9000
MINIO_KEY = os.environ["AWS_ACCESS_KEY_ID"]
MINIO_SECRET = os.environ["AWS_SECRET_ACCESS_KEY"]

BUCKET_CRUDO = "datasets-crudos"
BUCKET_GENERADO = "datasets-generados"

FILE_PATTERNS = {
    "details":    re.compile(r"StormEvents_details-.*\.csv$",    re.IGNORECASE),
    "locations":  re.compile(r"StormEvents_locations-.*\.csv$",  re.IGNORECASE),
    "fatalities": re.compile(r"StormEvents_fatalities-.*\.csv$", re.IGNORECASE),
}

# CPI-U promedio anual, base 2025. Fuente: https://www.in2013dollars.com/us-cpi
CPI_U = {
    1970: 38.8, 1971: 40.5, 1972: 41.8, 1973: 44.4, 1974: 49.3, 1975: 53.8,
    1976: 56.9, 1977: 60.6, 1978: 65.2, 1979: 72.6, 1980: 82.4, 1981: 90.9,
    1982: 96.5, 1983: 99.6, 1984: 103.9, 1985: 107.6, 1986: 109.6, 1987: 113.6,
    1988: 118.3, 1989: 124.0, 1990: 130.7, 1991: 136.2, 1992: 140.3, 1993: 144.5,
    1994: 148.2, 1995: 152.4, 1996: 156.9, 1997: 160.5, 1998: 163.0, 1999: 166.6,
    2000: 172.2, 2001: 177.1, 2002: 179.9, 2003: 184.0, 2004: 188.9, 2005: 195.3,
    2006: 201.6, 2007: 207.3, 2008: 215.3, 2009: 214.5, 2010: 218.1, 2011: 224.9,
    2012: 229.6, 2013: 233.0, 2014: 236.7, 2015: 237.0, 2016: 240.0, 2017: 245.1,
    2018: 251.1, 2019: 255.7, 2020: 258.8, 2021: 271.0, 2022: 292.7, 2023: 304.7,
    2024: 313.7, 2025: 322.1, 2026: 322.1,
}
CPI_BASE = CPI_U[2025]

# Región climática aproximada a partir del estado.
REGION_MAP = {
    # Tornado Alley
    "TEXAS": "Tornado Alley", "OKLAHOMA": "Tornado Alley", "KANSAS": "Tornado Alley",
    "NEBRASKA": "Tornado Alley", "SOUTH DAKOTA": "Tornado Alley",
    # Dixie Alley / Sureste
    "MISSISSIPPI": "Southeast", "ALABAMA": "Southeast", "GEORGIA": "Southeast",
    "TENNESSEE": "Southeast", "ARKANSAS": "Southeast", "LOUISIANA": "Southeast",
    "FLORIDA": "Southeast", "SOUTH CAROLINA": "Southeast", "NORTH CAROLINA": "Southeast",
    "KENTUCKY": "Southeast",
    # Medio Oeste
    "MISSOURI": "Midwest", "IOWA": "Midwest", "ILLINOIS": "Midwest",
    "INDIANA": "Midwest", "OHIO": "Midwest", "MICHIGAN": "Midwest",
    "WISCONSIN": "Midwest", "MINNESOTA": "Midwest", "NORTH DAKOTA": "Midwest",
    # Noreste
    "NEW YORK": "Northeast", "PENNSYLVANIA": "Northeast", "NEW JERSEY": "Northeast",
    "MASSACHUSETTS": "Northeast", "CONNECTICUT": "Northeast", "MAINE": "Northeast",
    "NEW HAMPSHIRE": "Northeast", "VERMONT": "Northeast", "RHODE ISLAND": "Northeast",
    "MARYLAND": "Northeast", "DELAWARE": "Northeast", "VIRGINIA": "Northeast",
    "WEST VIRGINIA": "Northeast", "DISTRICT OF COLUMBIA": "Northeast",
    # Oeste / Montañas
    "COLORADO": "West", "WYOMING": "West", "MONTANA": "West", "IDAHO": "West",
    "UTAH": "West", "NEVADA": "West", "ARIZONA": "West", "NEW MEXICO": "West",
    "CALIFORNIA": "West Coast", "OREGON": "West Coast", "WASHINGTON": "West Coast",
    # No contiguos
    "ALASKA": "Non-Contiguous", "HAWAII": "Non-Contiguous",
}

# Columnas del dataset limpio maestro.
MASTER_COLUMNS = [
    # Identificadores para auditoría y splits por episodio
    "EVENT_ID", "EPISODE_ID",
    # Contexto del evento
    "EVENT_TYPE", "CZ_TYPE", "STATE", "REGION", "WFO", "SOURCE",
    # Tiempo
    "YEAR", "MONTH", "HOUR", "DAY_OF_WEEK", "DAY_OF_YEAR", "IS_WEEKEND",
    "DECADE", "SEASON", "TIME_OF_DAY", "DURATION_MIN", "POST_1996",
    # Geografía: originales + derivadas
    "BEGIN_LAT", "BEGIN_LON", "END_LAT", "END_LON",
    "LAT_BIN", "LON_BIN", "TRACK_DISTANCE_KM",
    "HAS_COORDINATES", "HAS_TRACK_DISTANCE",
    # Magnitud y variables físicas
    "MAGNITUDE", "MAGNITUDE_TYPE", "HAS_MAGNITUDE",
    "FLOOD_CAUSE",
    "TOR_SCALE_NUM", "TOR_LENGTH", "TOR_WIDTH", "TOR_AREA_KM2", "HAS_TORNADO_DATA",
    # Consecuencias humanas: conservar para EDA; evaluar exclusión como predictores
    "TOTAL_DEATHS", "TOTAL_INJURIES", "TOTAL_CASUALTIES",
    "HAS_FATALITIES", "HAS_CASUALTIES",
    # Targets económicos y trazabilidad
    "DAMAGE_REAL_2025", "DAMAGE_CLASS",
]


# ---------------------------------------------------------------------------
# 2. Descubrimiento y carga de archivos crudos (desde MinIO)
# ---------------------------------------------------------------------------

def obtener_fs():
    """Crea la conexión a MinIO vía s3fs, reutilizable en todo el script."""
    return s3fs.S3FileSystem(
        key=MINIO_KEY,
        secret=MINIO_SECRET,
        client_kwargs={"endpoint_url": MINIO_ENDPOINT},
    )


def discover_files(fs, bucket):
    """Devuelve un dict {tipo: [keys]} escaneando TODO el bucket por nombre de
    archivo — no asume ninguna estructura particular de subcarpetas, así
    funciona sin importar cómo hayas organizado la carga al bucket."""
    found = {k: [] for k in FILE_PATTERNS}
    for key in fs.find(bucket):
        nombre = key.rsplit("/", 1)[-1]
        for kind, pattern in FILE_PATTERNS.items():
            if pattern.search(nombre):
                found[kind].append(key)
                break
    for kind, keys in found.items():
        print(f"{kind:<11}: {len(keys)} archivo(s)")
    return found


def load_concat(fs, keys, label):
    """Concatena una lista de CSV (leídos desde el bucket) reportando filas por archivo."""
    frames = []
    for key in sorted(keys):
        with fs.open(key) as f:
            df = pd.read_csv(f, low_memory=False)
        frames.append(df)
        print(f"  {key.rsplit('/', 1)[-1]}: {len(df):,} filas")
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    print(f"{label} total: {len(out):,} filas\n")
    return out


def cargar_datos_crudos(fs, bucket=BUCKET_CRUDO):
    """Descubre y concatena los tres tipos de archivo, leyendo desde el bucket."""
    files = discover_files(fs, bucket)
    print("=== DETAILS ===")
    details = load_concat(fs, files["details"], "details")
    print("=== LOCATIONS ===")
    locations = load_concat(fs, files["locations"], "locations")
    print("=== FATALITIES ===")
    fatalities = load_concat(fs, files["fatalities"], "fatalities")
    return details, locations, fatalities


# ---------------------------------------------------------------------------
# 3. Limpieza y enriquecimiento
# ---------------------------------------------------------------------------

def parse_damage(value):
    """Convierte montos de daño de NOAA a float.

    Distingue tres casos:
      - vacío / no reportado   -> np.nan  (FALTANTE genuino)
      - cero explícito ('0')   -> 0.0     (SIN daño)
      - valor con sufijo K/M/B -> float escalado
    """
    if pd.isna(value):
        return np.nan
    s = str(value).strip().upper().replace(",", "")
    if s in ("", "NAN", "NONE"):
        return np.nan
    if s in ("0", "0.0", "0.00", "0K", "0M", "0B"):
        return 0.0
    multipliers = {"K": 1e3, "M": 1e6, "B": 1e9, "T": 1e12, "H": 1e2}
    if s[-1] in multipliers:
        try:
            return float(s[:-1]) * multipliers[s[-1]]
        except ValueError:
            return np.nan
    try:
        return float(s)
    except ValueError:
        return np.nan


def build_datetime(df, prefix):
    """Reconstruye datetime desde columnas numéricas del CSV de NOAA."""
    ym = df[f"{prefix}_YEARMONTH"].astype("Int64").astype(str)
    day = df[f"{prefix}_DAY"].astype("Int64").astype(str).str.zfill(2)
    t = df[f"{prefix}_TIME"].astype("Int64").astype(str).str.zfill(4)
    year = ym.str[:4]
    month = ym.str[4:6]
    return pd.to_datetime(
        year + "-" + month + "-" + day + " " + t.str[:2] + ":" + t.str[2:],
        format="%Y-%m-%d %H:%M", errors="coerce"
    )


def limpiar_details(details):
    """Reconstruye fechas, parsea columnas de daño y elimina duplicados por EVENT_ID."""
    details = details.copy()
    details["BEGIN_DATE_TIME"] = build_datetime(details, "BEGIN")
    details["END_DATE_TIME"] = build_datetime(details, "END")
    details["YEAR"] = details["BEGIN_DATE_TIME"].dt.year
    details["MONTH"] = details["BEGIN_DATE_TIME"].dt.month

    for col in ("DAMAGE_PROPERTY", "DAMAGE_CROPS"):
        if col in details.columns:
            details[f"{col}_USD"] = details[col].apply(parse_damage)
            details[f"{col}_REPORTED"] = details[f"{col}_USD"].notna()

    before = len(details)
    details = details.drop_duplicates(subset="EVENT_ID", keep="first")
    print(f"Duplicados eliminados en details: {before - len(details):,}")
    print(f"Eventos únicos: {len(details):,}")

    for col in ("DAMAGE_PROPERTY_USD", "DAMAGE_CROPS_USD"):
        n_nan = details[col].isna().sum()
        n_zero = (details[col] == 0).sum()
        print(f"{col}: faltantes={n_nan:,} ({n_nan/len(details)*100:.1f}%) | "
              f"ceros explícitos={n_zero:,} ({n_zero/len(details)*100:.1f}%)")
    return details


# ---------------------------------------------------------------------------
# 4. EDA (cada función devuelve la figura y/o el DataFrame de resumen)
# ---------------------------------------------------------------------------

def eda_vista_general(details):
    """Chequeo de fechas sin parsear, rango temporal, y fechas futuras erróneas."""
    print("NaT en BEGIN:", details["BEGIN_DATE_TIME"].isna().sum())
    print("NaT en END:  ", details["END_DATE_TIME"].isna().sum())
    print(details["BEGIN_DATE_TIME"].dt.year.value_counts().sort_index().head(20))
    print(details["BEGIN_DATE_TIME"].dt.year.value_counts().sort_index().tail(20))

    futuros = details[details["BEGIN_DATE_TIME"] > "2026-12-31"]
    print(f"\nEventos con fecha futura: {len(futuros):,}")
    print(futuros[["EVENT_ID", "YEAR", "BEGIN_DATE_TIME"]].head(10))

    print("Rango temporal:",
          details["BEGIN_DATE_TIME"].min(), "→", details["BEGIN_DATE_TIME"].max())
    print(f"Tipos de evento únicos: {details['EVENT_TYPE'].nunique()}")
    print(f"Estados cubiertos: {details['STATE'].nunique()}")


def eda_eventos_por_anio(details):
    yearly = details.groupby("YEAR").size()
    fig, ax = plt.subplots(figsize=(12, 4))
    yearly.plot(ax=ax, color="steelblue")
    ax.set_title("Eventos registrados por año")
    ax.set_xlabel("Año"); ax.set_ylabel("Cantidad de eventos")
    plt.tight_layout()
    return fig


def eda_top_tipos_evento(details, top_n=15):
    top_types = details["EVENT_TYPE"].value_counts().head(top_n)
    fig, ax = plt.subplots(figsize=(10, 6))
    top_types.sort_values().plot.barh(ax=ax, color="darkorange")
    ax.set_title("Tipos de evento más frecuentes")
    ax.set_xlabel("Cantidad")
    plt.tight_layout()
    return fig


def eda_estacionalidad(details):
    monthly = details.groupby("MONTH").size()
    fig, ax = plt.subplots(figsize=(10, 4))
    monthly.plot.bar(ax=ax, color="teal")
    ax.set_title("Distribución mensual de eventos")
    ax.set_xlabel("Mes"); ax.set_ylabel("Cantidad")
    plt.tight_layout()
    return fig


def eda_danios_por_tipo(details, top_n=10):
    damage = (details.groupby("EVENT_TYPE")[["DAMAGE_PROPERTY_USD", "DAMAGE_CROPS_USD"]]
                     .sum()
                     .assign(TOTAL=lambda d: d.sum(axis=1))
                     .sort_values("TOTAL", ascending=False)
                     .head(top_n))
    damage_b = damage / 1e9  # en miles de millones de USD

    fig, ax = plt.subplots(figsize=(10, 6))
    damage_b[["DAMAGE_PROPERTY_USD", "DAMAGE_CROPS_USD"]].plot.barh(
        stacked=True, ax=ax, color=["#c0392b", "#27ae60"])
    ax.set_title("Daños acumulados por tipo de evento (USD miles de millones)")
    ax.set_xlabel("USD (miles de millones)")
    ax.legend(["Propiedad", "Cultivos"])
    plt.tight_layout()
    return fig, damage_b


def eda_victimas_por_tipo(details, top_n=10):
    victim_cols = ["DEATHS_DIRECT", "DEATHS_INDIRECT", "INJURIES_DIRECT", "INJURIES_INDIRECT"]
    victims = (details.groupby("EVENT_TYPE")[victim_cols]
                      .sum()
                      .assign(TOTAL_DEATHS=lambda d: d["DEATHS_DIRECT"] + d["DEATHS_INDIRECT"])
                      .sort_values("TOTAL_DEATHS", ascending=False)
                      .head(top_n))
    return victims


def eda_distribucion_geografica(details, top_n=15):
    by_state = details["STATE"].value_counts().head(top_n)
    fig, ax = plt.subplots(figsize=(10, 6))
    by_state.sort_values().plot.barh(ax=ax, color="slategray")
    ax.set_title("Estados con más eventos registrados")
    plt.tight_layout()
    return fig


def eda_faltantes(details):
    missing = pd.DataFrame({
        "n_missing": details.isna().sum(),
        "pct_missing": (details.isna().mean() * 100).round(2),
        "dtype": details.dtypes.astype(str)
    })
    missing = missing[missing["n_missing"] > 0].sort_values("pct_missing", ascending=False)
    print(f"Columnas con faltantes: {len(missing)} de {details.shape[1]}")

    fig, ax = plt.subplots(figsize=(10, max(4, len(missing) * 0.25)))
    missing["pct_missing"].sort_values().plot.barh(ax=ax, color="indianred")
    ax.set_title("Porcentaje de valores faltantes por columna")
    ax.set_xlabel("% faltante")
    ax.axvline(50, color="gray", linestyle="--", linewidth=0.8, label="50%")
    ax.legend()
    plt.tight_layout()
    return missing, fig


def eda_faltantes_tornados(details):
    tor_cols = [c for c in details.columns if c.startswith("TOR_")]
    tornados = details[details["EVENT_TYPE"] == "Tornado"]
    print(f"Eventos tipo Tornado: {len(tornados):,}")
    resumen = (tornados[tor_cols].isna().mean() * 100).round(2).sort_values(ascending=False)
    print("\nFaltantes en campos TOR_ dentro de tornados:")
    print(resumen)
    return resumen


def eda_outliers(details):
    numeric_cols = ["INJURIES_DIRECT", "INJURIES_INDIRECT",
                    "DEATHS_DIRECT", "DEATHS_INDIRECT",
                    "DAMAGE_PROPERTY_USD", "DAMAGE_CROPS_USD",
                    "MAGNITUDE", "TOR_LENGTH", "TOR_WIDTH"]

    summary = details[numeric_cols].describe(percentiles=[.5, .9, .95, .99]).T
    summary["skew"] = details[numeric_cols].skew()

    def iqr_outliers(series):
        s = series.dropna()
        q1, q3 = s.quantile([0.25, 0.75])
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n = ((s < lo) | (s > hi)).sum()
        return pd.Series({"n_outliers": n,
                          "pct_outliers": round(n / len(s) * 100, 2),
                          "low_bound": lo, "high_bound": hi})

    outlier_report = pd.DataFrame({c: iqr_outliers(details[c]) for c in numeric_cols}).T
    return summary, outlier_report


def eda_eventos_extremos(details, top_n=10):
    cols_show = ["BEGIN_DATE_TIME", "STATE", "EVENT_TYPE",
                 "DAMAGE_PROPERTY_USD", "DEATHS_DIRECT", "INJURIES_DIRECT"]
    top_damage = details.nlargest(top_n, "DAMAGE_PROPERTY_USD")[cols_show]
    top_deaths = details.nlargest(top_n, "DEATHS_DIRECT")[cols_show]
    return top_damage, top_deaths


def eda_boxplots_log(details):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    dmg = details.loc[details["DAMAGE_PROPERTY_USD"] > 0, "DAMAGE_PROPERTY_USD"]
    axes[0].boxplot(np.log10(dmg), vert=True)
    axes[0].set_title("Daño a propiedad (log10 USD, eventos > 0)")
    axes[0].set_ylabel("log10(USD)")

    mag = details["MAGNITUDE"].dropna()
    axes[1].boxplot(mag, vert=True)
    axes[1].set_title("Magnitud (viento en nudos / granizo en pulgadas)")

    plt.tight_layout()
    return fig


def eda_faltantes_danio(details):
    """Distingue faltante (no reportado) vs. cero explícito en las columnas de daño."""
    for col in ["DAMAGE_PROPERTY", "DAMAGE_CROPS"]:
        usd = details[f"{col}_USD"]
        n_missing = usd.isna().sum()
        n_zero = (usd == 0).sum()
        n_valid = (usd > 0).sum()
        print(f"{col}:")
        print(f"  No reportado (NaN) : {n_missing:>9,} ({n_missing/len(details)*100:5.1f}%)")
        print(f"  Cero explícito     : {n_zero:>9,} ({n_zero/len(details)*100:5.1f}%)")
        print(f"  Valor > 0          : {n_valid:>9,} ({n_valid/len(details)*100:5.1f}%)\n")

    print("Chequeo de coherencia con sección 8:")
    for col in ["DAMAGE_PROPERTY_USD", "DAMAGE_CROPS_USD"]:
        print(f"  {col}: {details[col].isna().sum():,} NaN "
              f"({details[col].isna().mean()*100:.1f}%)")


def eda_evolucion_faltante_danio(details):
    tmp = details.assign(_missing=details["DAMAGE_PROPERTY_USD"].isna())
    missing_by_year = tmp.groupby("YEAR")["_missing"].mean() * 100
    fig, ax = plt.subplots(figsize=(12, 4))
    missing_by_year.plot(ax=ax, color="crimson", marker="o", markersize=3)
    ax.set_title("% de eventos sin reporte de daño a la propiedad, por año")
    ax.set_xlabel("Año"); ax.set_ylabel("% faltante")
    ax.axhline(50, color="gray", linestyle="--", linewidth=0.8)
    plt.tight_layout()
    return fig


def eda_faltante_por_tipo_evento(details, top_n=15):
    tmp = details.assign(_missing=details["DAMAGE_PROPERTY_USD"].isna())
    missing_by_type = (tmp.groupby("EVENT_TYPE")["_missing"]
                          .agg(pct_missing="mean", n="size"))
    missing_by_type["pct_missing"] *= 100
    missing_by_type = missing_by_type.sort_values("pct_missing", ascending=False)
    return missing_by_type


# ---------------------------------------------------------------------------
# 5. Merge con locations y fatalities
# ---------------------------------------------------------------------------

def merge_geo_fatalities(details, locations, fatalities):
    """Vincula los tres DataFrames mediante EVENT_ID (formato largo, no 1:1)."""
    details_geo = details.merge(
        locations[["EVENT_ID", "LATITUDE", "LONGITUDE", "LOCATION"]],
        on="EVENT_ID", how="left")

    details_fat = details.merge(
        fatalities[["EVENT_ID", "FATALITY_TYPE", "FATALITY_AGE", "FATALITY_SEX", "FATALITY_LOCATION"]],
        on="EVENT_ID", how="inner")

    print(f"details_geo: {len(details_geo):,} filas (eventos × ubicaciones)")
    print(f"details_fat: {len(details_fat):,} filas (solo eventos con víctimas)")
    return details_geo, details_fat


# ---------------------------------------------------------------------------
# 6. Feature engineering determinístico
# ---------------------------------------------------------------------------

def get_season(month):
    if month in (12, 1, 2):
        return "Winter"
    if month in (3, 4, 5):
        return "Spring"
    if month in (6, 7, 8):
        return "Summer"
    return "Fall"


def time_of_day(hour):
    if pd.isna(hour):
        return np.nan
    if 0 <= hour < 6:
        return "Night"
    if 6 <= hour < 12:
        return "Morning"
    if 12 <= hour < 18:
        return "Afternoon"
    return "Evening"


def agregar_features_temporales(df):
    df["HOUR"] = df["BEGIN_DATE_TIME"].dt.hour
    df["DAY_OF_WEEK"] = df["BEGIN_DATE_TIME"].dt.dayofweek  # 0 = lunes
    df["DAY_OF_YEAR"] = df["BEGIN_DATE_TIME"].dt.dayofyear
    df["IS_WEEKEND"] = df["DAY_OF_WEEK"].isin([5, 6]).astype(int)
    df["DECADE"] = (df["YEAR"] // 10 * 10).astype("Int64")
    df["SEASON"] = df["MONTH"].apply(get_season)
    df["TIME_OF_DAY"] = df["HOUR"].apply(time_of_day)

    # Duración original en minutos. No se aplica clipping global: debe aprenderse con train.
    duration = (df["END_DATE_TIME"] - df["BEGIN_DATE_TIME"]).dt.total_seconds() / 60
    df["DURATION_MIN"] = duration.where(duration >= 0, np.nan)

    # NOAA amplió la cobertura de eventos a partir de 1996.
    df["POST_1996"] = (df["YEAR"] >= 1996).astype(int)

    print("Features temporales agregadas:")
    print(df[["YEAR", "MONTH", "HOUR", "DAY_OF_WEEK", "IS_WEEKEND", "SEASON",
              "TIME_OF_DAY", "DURATION_MIN", "DECADE", "POST_1996"]].head())
    return df


def haversine_km(lat1, lon1, lat2, lon2):
    """Distancia haversine entre dos puntos, en kilómetros."""
    radius_km = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * radius_km * np.arcsin(np.sqrt(a))


def agregar_features_geograficas(df):
    df["REGION"] = df["STATE"].map(REGION_MAP).fillna("Other/Marine")

    track_mask = df[["BEGIN_LAT", "BEGIN_LON", "END_LAT", "END_LON"]].notna().all(axis=1)
    df["TRACK_DISTANCE_KM"] = np.nan
    df.loc[track_mask, "TRACK_DISTANCE_KM"] = haversine_km(
        df.loc[track_mask, "BEGIN_LAT"], df.loc[track_mask, "BEGIN_LON"],
        df.loc[track_mask, "END_LAT"], df.loc[track_mask, "END_LON"]
    )

    # Flags de disponibilidad: preservan la diferencia entre cero real y dato faltante.
    df["HAS_COORDINATES"] = df[["BEGIN_LAT", "BEGIN_LON"]].notna().all(axis=1).astype(int)
    df["HAS_TRACK_DISTANCE"] = df["TRACK_DISTANCE_KM"].notna().astype(int)

    # Grid geográfico de 1°×1° para reducir cardinalidad sin eliminar coordenadas originales.
    df["LAT_BIN"] = df["BEGIN_LAT"].round(0)
    df["LON_BIN"] = df["BEGIN_LON"].round(0)

    print("Distribución por región:")
    print(df["REGION"].value_counts())
    return df


def parse_fscale(value):
    """Severidad ordinal del tornado a partir de TOR_F_SCALE (ej. 'EF3' -> 3)."""
    if pd.isna(value):
        return np.nan
    text = str(value).upper().replace("EF", "").replace("F", "").strip()
    try:
        return int(text)
    except ValueError:
        return np.nan


def zscore_within_group(series):
    std = series.std(ddof=0)
    if std == 0 or series.notna().sum() < 2:
        return pd.Series(np.nan, index=series.index)
    return (series - series.mean()) / std


def agregar_features_fisicas(df):
    """Agrega features físicas. Devuelve (df, magnitude_zscore_eda).

    magnitude_zscore_eda se calcula SOLO para EDA con estadísticas de todo el
    dataset — no se exporta ni se agrega a `df`, porque debe recalcularse con
    estadísticas aprendidas únicamente en train al momento de modelar.
    """
    df["TOR_SCALE_NUM"] = df["TOR_F_SCALE"].apply(parse_fscale)

    # Área aproximada barrida por el tornado (millas × yardas → km²). No aplica a no-tornados.
    tornado_mask = df["EVENT_TYPE"].eq("Tornado")
    df["TOR_AREA_KM2"] = np.nan
    df.loc[tornado_mask, "TOR_AREA_KM2"] = (
        df.loc[tornado_mask, "TOR_LENGTH"] * 1.60934
        * df.loc[tornado_mask, "TOR_WIDTH"] * 0.0009144
    )

    df["HAS_MAGNITUDE"] = df["MAGNITUDE"].notna().astype(int)
    df["HAS_TORNADO_DATA"] = df[["TOR_SCALE_NUM", "TOR_LENGTH", "TOR_WIDTH"]].notna().any(axis=1).astype(int)

    magnitude_zscore_eda = (
        df.groupby("EVENT_TYPE")["MAGNITUDE"]
          .transform(zscore_within_group)
    )

    print("Resumen de features físicas exportables:")
    print(df[["MAGNITUDE", "TOR_SCALE_NUM", "TOR_LENGTH", "TOR_WIDTH", "TOR_AREA_KM2",
              "HAS_MAGNITUDE", "HAS_TORNADO_DATA"]].describe())
    print("\nMAGNITUDE_ZSCORE temporal para EDA (no exportado):")
    print(magnitude_zscore_eda.describe())
    return df, magnitude_zscore_eda


def agregar_variables_impacto(df):
    """Variables de consecuencias humanas y monto de daño total.

    OJO: si estas variables se usan como features en algún problema relacionado,
    asegurate de que el target de ese problema no las contenga (leakage circular).
    """
    df["TOTAL_DEATHS"] = df["DEATHS_DIRECT"] + df["DEATHS_INDIRECT"]
    df["TOTAL_INJURIES"] = df["INJURIES_DIRECT"] + df["INJURIES_INDIRECT"]
    df["TOTAL_CASUALTIES"] = df["TOTAL_DEATHS"] + df["TOTAL_INJURIES"]

    # Daño total: se PRESERVAN los faltantes (no se aplastan a 0).
    prop = df["DAMAGE_PROPERTY_USD"]
    crop = df["DAMAGE_CROPS_USD"]
    df["DAMAGE_REPORTED"] = prop.notna() | crop.notna()
    df["TOTAL_DAMAGE_USD"] = prop.add(crop, fill_value=0)
    df.loc[~df["DAMAGE_REPORTED"], "TOTAL_DAMAGE_USD"] = np.nan

    df["HAS_FATALITIES"] = (df["TOTAL_DEATHS"] > 0).astype(int)
    df["HAS_CASUALTIES"] = (df["TOTAL_CASUALTIES"] > 0).astype(int)
    df["IS_VIOLENT_TORNADO"] = (df["TOR_SCALE_NUM"] >= 3).astype("Int64")
    df["IS_MAJOR_DAMAGE"] = np.where(
        df["DAMAGE_REPORTED"], (df["TOTAL_DAMAGE_USD"] >= 1_000_000).astype(float), np.nan
    )

    print("Eventos con al menos una componente de daño reportada:")
    print(f"  {df['DAMAGE_REPORTED'].sum():,} / {len(df):,} "
          f"({df['DAMAGE_REPORTED'].mean() * 100:.2f}%)")
    return df


def resumen_tasas_base(df):
    """Imprime tasas base de potenciales targets (fatalities, casualties, major damage)."""
    print("Tasas base de potenciales targets:")
    for col in ["HAS_FATALITIES", "HAS_CASUALTIES"]:
        print(f"  {col:20s}: {df[col].mean()*100:.2f}%")

    rep = df["DAMAGE_REPORTED"]
    print(f"  IS_MAJOR_DAMAGE (entre reportados): "
          f"{df.loc[rep, 'IS_MAJOR_DAMAGE'].mean()*100:.2f}% "
          f"(reportados: {rep.sum():,} / {len(df):,})")

    tornado_mask = df["HAS_TORNADO_DATA"] == 1
    print(f"  IS_VIOLENT_TORNADO (entre tornados): "
          f"{df.loc[tornado_mask, 'IS_VIOLENT_TORNADO'].mean()*100:.2f}%")


# ---------------------------------------------------------------------------
# 7. Ajuste por inflación y construcción del target
# ---------------------------------------------------------------------------

def ajustar_por_inflacion(df):
    """Agrega DAMAGE_REAL_2025: daño histórico convertido a USD comparables de 2025."""
    year = df["YEAR"] if "YEAR" in df.columns else df["BEGIN_DATE_TIME"].dt.year
    cpi = year.map(CPI_U).fillna(CPI_BASE)
    df["DAMAGE_REAL_2025"] = df["TOTAL_DAMAGE_USD"] * (CPI_BASE / cpi)

    mask = df["TOTAL_DAMAGE_USD"] > 0
    print(df.loc[mask, ["TOTAL_DAMAGE_USD", "DAMAGE_REAL_2025"]]
          .describe(percentiles=[.5, .9, .99]).round(0))
    return df


def construir_target_damage_class(df):
    """Construye DAMAGE_CLASS: target multiclase (0-4) de nivel de daño económico.

    Se usa una definición explícita para que la clase 0 contenga exclusivamente
    daño = 0 (no confundir con "no reportado", que queda NaN).
    """
    damage = df["DAMAGE_REAL_2025"]

    conditions = [
        damage.eq(0),
        damage.gt(0) & damage.le(1e4),
        damage.gt(1e4) & damage.le(1e5),
        damage.gt(1e5) & damage.le(1e6),
        damage.gt(1e6),
    ]
    labels = [0, 1, 2, 3, 4]

    df["DAMAGE_CLASS"] = pd.Series(
        np.select(conditions, labels, default=np.nan),
        index=df.index,
        dtype="Float64"
    ).astype("Int64")

    class_names = {
        0: "Sin daño (= 0)",
        1: "Menor (0-10K]",
        2: "Moderado (10K-100K]",
        3: "Severo (100K-1M]",
        4: "Catastrófico (>1M)",
    }

    modeling_mask = df["DAMAGE_REAL_2025"].notna()
    print(f"Eventos con daño reportado: {modeling_mask.sum():,}")
    print(f"Eventos sin daño reportado: {(~modeling_mask).sum():,}")
    print("\nDistribución del target entre eventos con daño reportado:")
    dist = df.loc[modeling_mask, "DAMAGE_CLASS"].value_counts().sort_index()
    for cls, count in dist.items():
        print(f"  {cls} - {class_names[int(cls)]:24s}: {count:>9,} "
              f"({count / modeling_mask.sum() * 100:.2f}%)")
    return df


# ---------------------------------------------------------------------------
# 8. Selección de columnas, chequeos de coherencia y exportación
# ---------------------------------------------------------------------------

def seleccionar_columnas(df):
    """Arma el dataset maestro (todas las columnas relevantes) y la base para
    modelado de daño (solo eventos con DAMAGE_CLASS disponible)."""
    columns = [column for column in MASTER_COLUMNS if column in df.columns]
    storm_events_clean_master = df[columns].copy()
    storm_events_damage_modeling_base = storm_events_clean_master[
        storm_events_clean_master["DAMAGE_CLASS"].notna()
    ].reset_index(drop=True)

    print(f"Dataset maestro: {storm_events_clean_master.shape[0]:,} filas × "
          f"{storm_events_clean_master.shape[1]} columnas")
    print(f"Base para modelado de daño: {storm_events_damage_modeling_base.shape[0]:,} filas × "
          f"{storm_events_damage_modeling_base.shape[1]} columnas")
    print("\nColumnas exportadas:")
    for index, column in enumerate(storm_events_clean_master.columns, start=1):
        print(f"  {index:2d}. {column} ({storm_events_clean_master[column].dtype})")

    return storm_events_clean_master, storm_events_damage_modeling_base


def validar_dataset(storm_events_clean_master, storm_events_damage_modeling_base):
    """Chequeos de coherencia antes de exportar. Lanza AssertionError si algo falla."""
    assert storm_events_clean_master["EVENT_ID"].is_unique, "EVENT_ID debe ser único."
    assert storm_events_damage_modeling_base["DAMAGE_CLASS"].notna().all()
    assert storm_events_damage_modeling_base["DAMAGE_REAL_2025"].notna().all()
    assert storm_events_damage_modeling_base.loc[
        storm_events_damage_modeling_base["DAMAGE_CLASS"] == 0,
        "DAMAGE_REAL_2025"
    ].eq(0).all(), "La clase 0 debe contener únicamente daño exactamente igual a cero."
    assert "MAGNITUDE_ZSCORE" not in storm_events_clean_master.columns
    assert "IS_MARINE" not in storm_events_clean_master.columns
    assert "IS_ZONE_FORECAST" not in storm_events_clean_master.columns

    print("Chequeos superados correctamente.")


def exportar_datasets(fs, storm_events_clean_master, storm_events_damage_modeling_base,
                       bucket=BUCKET_GENERADO):
    """Exporta el dataset maestro y la base de modelado como CSV directo al bucket."""
    master_key = f"{bucket}/storm_events_clean_master.csv"
    modeling_key = f"{bucket}/storm_events_damage_modeling_base.csv"

    with fs.open(master_key, "w") as f:
        storm_events_clean_master.to_csv(f, index=False)
    with fs.open(modeling_key, "w") as f:
        storm_events_damage_modeling_base.to_csv(f, index=False)

    print(f"Exportado: s3://{master_key} ({storm_events_clean_master.shape[0]:,} filas × "
          f"{storm_events_clean_master.shape[1]} columnas)")
    print(f"Exportado: s3://{modeling_key} ({storm_events_damage_modeling_base.shape[0]:,} filas × "
          f"{storm_events_damage_modeling_base.shape[1]} columnas)")

    return master_key, modeling_key


# ---------------------------------------------------------------------------
# 9. main
# ---------------------------------------------------------------------------

def main():
    fs = obtener_fs()

    details, locations, fatalities = cargar_datos_crudos(fs)
    details = limpiar_details(details)

    # --- EDA opcional: descomentar lo que quieras correr/loguear ---
    # eda_vista_general(details)
    # eda_eventos_por_anio(details)
    # eda_top_tipos_evento(details)
    # eda_estacionalidad(details)
    # eda_danios_por_tipo(details)
    # eda_victimas_por_tipo(details)
    # eda_distribucion_geografica(details)
    # eda_faltantes(details)
    # eda_faltantes_tornados(details)
    # eda_outliers(details)
    # eda_eventos_extremos(details)
    # eda_boxplots_log(details)
    # eda_faltantes_danio(details)
    # eda_evolucion_faltante_danio(details)
    # eda_faltante_por_tipo_evento(details)

    details_geo, details_fat = merge_geo_fatalities(details, locations, fatalities)

    df = details.copy()
    print(f"Filas: {len(df):,} | Columnas iniciales: {df.shape[1]}")

    df = agregar_features_temporales(df)
    df = agregar_features_geograficas(df)
    df, magnitude_zscore_eda = agregar_features_fisicas(df)
    df = agregar_variables_impacto(df)
    resumen_tasas_base(df)

    df = ajustar_por_inflacion(df)
    df = construir_target_damage_class(df)

    storm_events_clean_master, storm_events_damage_modeling_base = seleccionar_columnas(df)
    validar_dataset(storm_events_clean_master, storm_events_damage_modeling_base)
    exportar_datasets(fs, storm_events_clean_master, storm_events_damage_modeling_base)


if __name__ == "__main__":
    main()
