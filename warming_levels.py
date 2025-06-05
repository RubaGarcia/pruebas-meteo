import pandas as pd
# import numpy as np
import xarray as xr
import math

def obtener_rangos_por_warming_level(csv_path, warming_level):
    """
    Extrae los rangos temporales para un warming level dado desde un archivo CSV.

    Args:
        csv_path (str): Ruta al archivo CSV.
        warming_level (float): Warming level (ej. 1.5, 2, 3, 4).

    Returns:
        dict: Diccionario con claves como 'ssp126', 'ssp245', etc. y valores con listas de periodos.
    """
    # Cargar CSV con múltiples encabezados
    df = pd.read_csv(csv_path, header=[0, 1], index_col=0)

    warming_level = float(warming_level)

    # Filtrar columnas del warming level deseado
    cols_deseadas = [col for col in df.columns if float(col[0]) == warming_level]

    resultado = {}

    for gcm, fila in df.iterrows():
        info_gcm = {}
        for col in cols_deseadas:
            escenario = col[1]
            valor = fila[col]
            if isinstance(valor, str) and ('NA' in valor or valor.strip() == ''):
                continue
            if valor == 9999 or valor == "9999":
                continue
            info_gcm[escenario] = valor
            # print(valor)
        if info_gcm:
            resultado[gcm.lower()] = info_gcm

    return resultado



def conditional_nanmean(ds, dim, nan_threshold=0.8):
    """
    Applies regular mean if proportion of NaNs <= threshold.
    If more NaNs, uses nanmean (ignores NaNs).
    """
    # Contar NaNs y total por pixel (lat, lon)
    isnan = ds.isnull()
    nan_fraction = isnan.sum(dim=dim) / ds.sizes[dim]

    # Más del 80% NaNs → usar nanmean (ignora NaNs)
    use_nanmean = nan_fraction > nan_threshold

    mean_normal = ds.mean(dim=dim, skipna=False)
    mean_ignore_nan = ds.mean(dim=dim, skipna=True)

    # Escoger valor apropiado en cada celda
    result = xr.where(use_nanmean, mean_ignore_nan, mean_normal)

    return result

if __name__ == "__main__":
    # Ejemplo de uso
    warming_levels = 1.5
    csv_warmign_levels = 'CMIP6_WarmingLevels.csv'


    rangos = obtener_rangos_por_warming_level(csv_warmign_levels, warming_levels)

    # obtain_time_margins(rangos)
