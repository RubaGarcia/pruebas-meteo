import numpy as np
import xarray as xr
import os
from termcolor import colored

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
        'rice': 4
    }
    crop_index=crop_mapping[crop_name]
    return ds.sel(crop=crop_index)



def select_data_for_warming_level_time_range(
    self,
    dataset: xr.Dataset,
    level: str,
    time_filter_range: str,
    aggregation_stat: str,
):
    """
    TODO:Acabar de arreglarlo y ver como funciona el método
    Select data from the given dataset for the time period according to warming levels.

    Parameters
    ----------
    dataset: xarray.Dataset
        The dataset containing the historical and future data.
    level: float
        The warming level to select data for.
    time_filter_range: slice
        The time range to filter the data with.
    aggregation_stat: str
        A string indicating the statistic applied as the aggregation function.

    Returns
    -------
    xarray.Dataset
        A dataset containing the selected data for the specified warming level
        and time range.
    """
    # Make static dataset to be filled. Same as "dataset" without time dim
    data_for_members = []
    data_for_members_clim = []
    # Iterate over the members inside the dataset
    for mem_id in range(len(dataset.member_id)):
        # Get the data for the member, but having the member dimension yet
        data_for_member = dataset.isel(member_id=slice(mem_id, mem_id + 1))

        # Get the member name
        member_name = str(data_for_member.member_id.values[0])

        period_range = self.get_time_range_when_warming_level_is_reached(
            warming_level=level,
            member=member_name,
        )

        # If empty, continue to next member
        if period_range == "":
            continue

        # If the value is 9999,
        # it indicates that warming level is not reached
        elif (
            period_range == "9999" or period_range == 9999 or period_range == 9999.0
        ):
            continue

        # The other option is the "time_period"
        # to be a string like "2016-2035"
        else:
            data_for_member, data_for_member_clim = TemporalAggregation(
                dataset=data_for_member,
                period_range=period_range,
                time_filter=time_filter_range,
                statistical=aggregation_stat,
                product_type=self.product_type,
            ).compute()
        # Assign data to the pre-generated dataset
        data_for_members.append(data_for_member)
        data_for_members_clim.append(data_for_member_clim)
    if len(data_for_members) == 0 or len(data_for_members_clim) == 0:
        raise EmptyDatasetListError()
    data_for_members = xr.concat(data_for_members, dim="member_id")
    data_for_members_clim = xr.xarray.concat(data_for_members_clim, dim="member_id")
    return data_for_members, data_for_members_clim


# os.system('color')



gcm_list=["gfdl-esm4","ipsl-cm6a-lr","mpi-esm1-2-hr","mri-esm2-0","ukesm1-0-ll"]
crop_model_list=[ "acea","crover","cygma1p74","dssat-pythia","epic-iiasa", "isam","ldndc", "lpjml", "pdssat", "pepic", "promet","simplace-lintul5"]
# Combine each element of gcm_list with each element of crop_model_list
combined_list = [f"{gcm}_{crop}" for gcm in gcm_list for crop in crop_model_list]
# combined_list = ["gfdl-esm4_crover"]



crop_list=["maize","wheat","soybean","rice"]
# Load the NetCDF file


experiment_list=["ssp126","ssp370","ssp585"]
experiment_list=["ssp126"]

# Define the periods with their start and end years
periods = {
    'near': {'slice': slice(2016, 2035), 'range': '2016-2035'},
    'medium': {'slice': slice(2046, 2065), 'range': '2046-2065'},
    'far': {'slice': slice(2081, 2100), 'range': '2081-2100'}
}

baseline =  "1983-2013"
root ="/home/adri/ISIMIP-simulations/www.pik-potsdam.de/~jonasjae"

for crop_name in crop_list:
    for experiment in experiment_list:
        member_data=[]
        for member in combined_list:
            try:
            
                file_path = f'{root}/datasets/GGCMI_Phase3_annual_{experiment}_{member}.nc4'
                if not os.path.exists(file_path):
                    print(colored('[SKIP]', 'red'),'Fichero no encontrado')
                    continue
                ds_all = xr.open_dataset(file_path)
                ds_crop=select_crop(ds_all, crop_name)
                # print(ds_all.coords['crop'])
                # Create a list to store the mean data for each period
                mean_data = []
                
                # print(ds_crop["yield change"].isel(years=0).values)
                # Calculate the time mean for each period
                for period_name, period_info in periods.items():
                    # Select the data for the current period
                    ds_period = ds_crop.sel(years=period_info['slice'])

                    # Validamos que el mapa no tenga min = max (sin variación)
                    yc_values = ds_period['yield change'].values

                    if np.isnan(yc_values).all():
                        print(colored(f'[SKIP] {member} - {period_name} tiene todo NaN en yield change', 'red'))
                        continue

                    # Calculate the time mean
                    ds_period_mean = ds_period.mean(dim='years', skipna=True)

                    # Add a 'period' coordinate with the period range
                    ds_period_mean = ds_period_mean.expand_dims({'period': [period_info['range']]})#TODO no vale skipna
                    # Add a 'baseline' coordinate with the baseline range
                    ds_period_mean = ds_period_mean.expand_dims({'baseline': [baseline]})
                    # Add a 'model' coordinate with the models ciombination 
                    ds_period_mean = ds_period_mean.expand_dims({'member': [member]})  
                    # Append to the list

                    

                    print(colored(member, 'yellow'))

                    mean_data.append(ds_period_mean)
            except Exception as e:
                print (e, member)
            
            # Concatenate the mean of each period data along the 'period' dimension
            # ds_mean_member
            if mean_data and len(mean_data) > 0:  # Si no está vacío
                ds_mean_member = xr.concat(mean_data, dim='period')
                member_data.append(ds_mean_member)
            else:
                print(colored(f'[SKIP] No hay datos válidos para el miembro {member} en el experimento {experiment} y cultivo {crop_name}', 'red'))
                continue

            # member_data.append(ds_mean_member)
        # Concatenate the mean data of each period and each member data along the 'member' dimension
        # print (member_data)
        if mean_data:
            ds_mean_all = xr.concat(member_data, dim='member')
        # Write the result to a new NetCDF file
        output_file_path = f'yield_climatology_{experiment}_{crop_name}.nc'
        if mean_data:


            ds_mean_all = xr.concat(member_data, dim='member')

            # Calcular la media general a lo largo de todos los miembros
            ds_mean_general = ds_mean_all.mean(dim='member', keep_attrs=True)

            # Añadir un identificador para la media general como un nuevo "miembro"
            ds_mean_general = ds_mean_general.expand_dims({'member': ['ensemble_mean']})

            # Opción 1: Insertar la media al inicio
            ds_mean_all = xr.concat([ds_mean_general, ds_mean_all], dim='member')

        anom_sign = xr.apply_ufunc(np.sign, ds_mean_all['yield change'])
        ensemble_sign = anom_sign.sel(member='ensemble_mean')
        members_sign = anom_sign.drop_sel(member='ensemble_mean')

        equal_sign = members_sign== ensemble_sign
        equal_sign = equal_sign.where(~np.isnan(members_sign))

        agreement_fraction = equal_sign.mean(dim='member', skipna=True)
        agreement_fraction = agreement_fraction.fillna(0)

        yield_consensus_array = xr.where(agreement_fraction < 0.8, 1.0, 0.0).astype('float32')

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

        # Asignar al dataset
        ds_mean_all['yield_consensus'] = consensus_da


        write_to_netcdf(ds_mean_all, output_file_path)

        print(colored(f"Data saved to {output_file_path}", 'green'))
        ds_all.close()
        # # Concatenate the mean of each period data along the 'period' dimension
        # if ds_mean_member:
        #     ds_mean_member = xr.concat(mean_data, dim='period')
        #     member_data.append(ds_mean_member)
    