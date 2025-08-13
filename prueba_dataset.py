import numpy as np
import xarray as xr
import os
from termcolor import colored

from warming_levels import obtener_rangos_por_warming_level, conditional_nanmean
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)


def write_to_netcdf(dataset: xr.Dataset, path: str, var: str = None) :
    """
    Save an xarray.Dataset as a netCDF file with custom encoding for compression and chunking.
    """

    encoding = {}
    for v in dataset.data_vars:
        var_dims = dataset[v].dims
        # Define chunk size of 1 for non-spatial dims, full size for lat/lon
        chunk_sizes = []
        for dim in var_dims:
            if dim in ['lat']:
                chunk_sizes.append(len(dataset.lat))
            elif dim in ['lon']:
                chunk_sizes.append(len(dataset.lon))
            else:
                chunk_sizes.append(1)  # chunk size 1 for member, period, baseline, etc.

        encoding[v] = dict(
            dtype="float32",
            shuffle=True,
            zlib=True,
            complevel=1,
            chunksizes=tuple(chunk_sizes)
        )

    dataset.to_netcdf(path=path)


def select_crop(ds, crop_name):
    # Mapping of crop names to their respective indices
    crop_mapping = {
        'maize': 1,
        'wheat': 2,
        'soybean': 3,
        'rice': 4,
        'millet': 5,
        'sorghum': 6,
        'cassava': 7,
        'pulses': 8,
        'potato': 9
    }
    crop_index=crop_mapping[crop_name]
    return ds.sel(crop=crop_index)

#call the warming levels function

# os.system('color')



gcm_list=["gfdl-esm4","ipsl-cm6a-lr","mpi-esm1-2-hr","mri-esm2-0","ukesm1-0-ll"]
crop_model_list=[ "acea","crover","cygma1p74","dssat-pythia","epic-iiasa", "isam","ldndc", "lpjml", "pdssat", "pepic", "promet","simplace-lintul5"]
# Combine each element of gcm_list with each element of crop_model_list
# combined_list = ["gfdl-esm4_crover"]
crop_model_list = ["cygma1p74", "lpj-guess", "lpjml", "simplace-lintul5"]



crop_list=["maize","wheat","soybean","rice", "millet","sorghum","cassava","pulses","potato"]
# Load the NetCDF file


experiment_list=["ssp126","ssp370","ssp585"]
# experiment_list=["ssp126"]

# Define the periods with their start and end years
periods = {
    'near': {'slice': slice(2016, 2035), 'range': '2016-2035'},
    'medium': {'slice': slice(2046, 2065), 'range': '2046-2065'},
    'far': {'slice': slice(2081, 2100), 'range': '2081-2100'},
    'warming_1.5': None,  # Placeholder for warming levels
    'warming_2.0': None,  # Placeholder for warming levels
    'warming_3.0': None,  # Placeholder for warming levels
}

baseline =  "1983-2013"

warming_levels = [1.5, 2.0, 3.0]
csv_warmign_levels = 'CMIP6_WarmingLevels.csv'

warming_dict = {
    wl: obtener_rangos_por_warming_level(csv_warmign_levels, wl) 
    for wl in warming_levels
}


combined_list = [f"{gcm}_{crop}" for gcm in gcm_list for crop in crop_model_list]


warming_values = {}

for warming_level, gcm_text_dict in warming_dict.items():
    for texto_gcm, experiment_dict in gcm_text_dict.items():
        for gcm in gcm_list:
            # Si el gcm es 'mpi-esm1-2-hr', buscar como 'mpi-esm1-2-lr' en texto_gcm
            search_gcm = 'mpi-esm1-2-lr' if gcm == 'mpi-esm1-2-hr' else gcm
            if search_gcm in texto_gcm:
                for experiment, valor in experiment_dict.items():
                    warming_values[(warming_level, gcm, experiment)] = valor

periods_gcm_experiment = {}

for gcm in gcm_list:
    for experiment in experiment_list:
        # Empezamos con los periodos fijos
        periods_gcm_experiment[(gcm, experiment)] = periods.copy()

        # Ahora agregamos los periodos warming
        for wl in warming_levels:
            valor = warming_values.get((wl, gcm, experiment), None)
            if valor is not None:
                # Aquí interpretamos 'valor' para construir el periodo warming
                # Por ejemplo, si 'valor' es un string tipo "2041-2060"
                try:
                    start_year, end_year = map(int, valor.split('-'))
                    periods_gcm_experiment[(gcm, experiment)][f'warming_{wl}'] = {
                        'slice': slice(start_year, end_year),
                        'range': f'GWL {wl}'
                    }
                except Exception as e:
                    # Si no puedes interpretar el valor, lo marcas o pones None
                    periods_gcm_experiment[(gcm, experiment)][f'warming_{wl}'] = None
            else:
                periods_gcm_experiment[(gcm, experiment)][f'warming_{wl}'] = None

for crop_name in crop_list:
    for experiment in experiment_list:
        member_data=[]
        for member in combined_list:
            try:
                # file_path = f'datasets/GGCMI_Phase3_annual_{experiment}_{member}.nc4'
                file_path = f'datasets/GGCMI_Phase3_annual_{experiment}_{member}_allcrops.nc4'
                if not os.path.exists(file_path):
                    print(colored('[SKIP]', 'red'), 'File not found')
                    continue

                ds_all = xr.open_dataset(file_path)
                
                ds_crop=select_crop(ds_all, crop_name)
                mean_data = []
                gcm = member.split('_')[0]

                periods_for_gcm_exp = periods_gcm_experiment.get((gcm, experiment), periods)

                for period_name, period_info in periods_for_gcm_exp.items():
                    if period_info is None:
                        continue

                    ds_period = ds_crop.sel(years=period_info['slice'])
                    yc_values = ds_period['yield change'].values

                    if np.isnan(yc_values).all():
                        print(colored(f'[SKIP] {member} - {period_name} has all NaN values in yield change', 'red'))
                        continue

                    # Obtener el eje de 'years' correctamente
                    years_axis = ds_period['yield change'].get_axis_num('years')
                    nan_fraction = np.isnan(yc_values).sum(axis=years_axis) / yc_values.shape[years_axis]
                    mask = nan_fraction > 0.2

                    ds_period_mean = ds_period.mean(dim='years', skipna=True)
                    ds_period_mean['yield change'] = ds_period_mean['yield change'].where(~mask)

                    ds_period_mean = ds_period_mean.expand_dims({'period': [period_info['range']]})
                    ds_period_mean = ds_period_mean.expand_dims({'baseline': [baseline]})
                    ds_period_mean = ds_period_mean.expand_dims({'member': [member]})
                    print(colored(member, 'yellow'))
                    mean_data.append(ds_period_mean)
                
            except Exception as e:
                print (e, member)
            
            if mean_data and len(mean_data) > 0:
                ds_mean_member = xr.concat(mean_data, dim='period')
                _, unique_idx = np.unique(ds_mean_member['period'].values, return_index=True)
                ds_mean_member = ds_mean_member.isel(period=sorted(unique_idx))
                member_data.append(ds_mean_member)
            else:
                print(colored(f'[SKIP] No valid data for member {member} in experiment {experiment} and crop {crop_name}', 'red'))
                continue

        output_file_path = f'files/yield_climatology_{experiment}_{crop_name}.nc'
        if mean_data:
            ds_mean_all = xr.concat(member_data, dim='member')
            ds_mean_general = ds_mean_all.mean(dim='member', keep_attrs=True)
            ds_mean_general = ds_mean_general.expand_dims({'member': ['ensemble_mean']})
            ds_mean_all = xr.concat([ds_mean_general, ds_mean_all], dim='member')
            if np.isnan(ds_mean_all['yield change'].values).all():
                print(colored(f'[SKIP] Empty map for {experiment} - {crop_name}', 'red'))
                continue
        anom_sign = xr.apply_ufunc(np.sign, ds_mean_all['yield change'])
        ensemble_sign = anom_sign.sel(member='ensemble_mean')
        members_sign = anom_sign.drop_sel(member='ensemble_mean')

        equal_sign = members_sign == ensemble_sign
        equal_sign = equal_sign.where(~np.isnan(members_sign))

        agreement_fraction = equal_sign.mean(dim='member', skipna=True)

        yield_consensus_array = xr.where(
            agreement_fraction < 0.8, 1.0, 0.0
        ).where(~np.isnan(agreement_fraction)).astype('float32')

        consensus_da = xr.DataArray(
            data=yield_consensus_array,
            dims=['baseline', 'period', 'lat', 'lon'],
            coords={
                'baseline': ds_mean_all['baseline'],
                'period': ds_mean_all['period'],
                'lat': ds_mean_all['lat'],
                'lon': ds_mean_all['lon']
            },
            name='yield_consensus',
            attrs={
                '_FillValue': np.nan,
                'description': 'Model agreement on relative change sign (less than 80% agreement = 1)',
                'long_name': 'Yield sign consensus disagreement'
            }
        )

        ds_mean_all['yield_anom'] = consensus_da
        
        write_to_netcdf(ds_mean_all, output_file_path)

        print(colored(f"Data saved to {output_file_path}", 'green'))
        ds_all.close()
