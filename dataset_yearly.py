import numpy as np
import xarray as xr
import os
from termcolor import colored
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

def write_to_netcdf(dataset: xr.Dataset, path: str):
    desired_order = ('lat', 'lon', 'member', 'time')
    for v in dataset.data_vars:
        dims = dataset[v].dims
        # Solo reordena si todas las dims están presentes
        if all(dim in dims for dim in desired_order):
            dataset[v] = dataset[v].transpose(*desired_order)
    encoding = {}
    n_members = dataset.dims.get('member', 1)
    for v in dataset.data_vars:
        encoding[v] = dict(
            dtype="float32",
            shuffle=True,
            zlib=True,
            complevel=5,
            chunksizes=(20, 20, n_members, 100)
        )
    dataset.to_netcdf(path=path, encoding=encoding, engine="netcdf4")

def select_crop(ds, crop_name):
    crop_mapping = {'maize': 1, 'wheat': 2, 'soybean': 3, 'rice': 4}
    return ds.sel(crop=crop_mapping[crop_name])

gcm_list = ["gfdl-esm4", "ipsl-cm6a-lr", "mpi-esm1-2-hr", "mri-esm2-0", "ukesm1-0-ll"]
crop_model_list = ["acea", "crover", "cygma1p74", "dssat-pythia", "epic-iiasa", "isam", "ldndc", "lpjml", "pdssat", "pepic", "promet", "simplace-lintul5"]
crop_list = ["maize", "wheat", "soybean", "rice"]
experiment_list = ["ssp126", "ssp370", "ssp585"]
combined_list = [f"{gcm}_{crop}" for gcm in gcm_list for crop in crop_model_list]

for crop_name in crop_list:
    for experiment in experiment_list:
        member_data = []

        for member in combined_list:
            file_path = f'datasets/GGCMI_Phase3_annual_{experiment}_{member}.nc4'
            if not os.path.exists(file_path):
                print(colored('[SKIP]', 'red'), 'File not found:', file_path)
                continue
            try:
                ds_all = xr.open_dataset(file_path)
                ds_crop = select_crop(ds_all, crop_name)
                if 'yield change' in ds_crop:
                    ds_crop = ds_crop.rename({'yield change': 'yield_relanom'})
                if 'yield_relanom' not in ds_crop:
                    print(colored('[SKIP]', 'red'), f"No 'yield_relanom' in {member}")
                    continue

                # Selecciona solo 'yield_relanom'
                ds_crop = ds_crop[['yield_relanom']]

                if np.isnan(ds_crop['yield_relanom'].values).all():
                    print(colored('[SKIP]', 'red'), f"All NaNs in {member}")
                    continue

                ds_crop = ds_crop.expand_dims({'member': [member]})
                member_data.append(ds_crop)

            except Exception as e:
                print(colored('[ERROR]', 'red'), member, e)
                continue

        if member_data:
            ds_all_members = xr.concat(member_data, dim='member')
            ensemble_mean = ds_all_members.mean(dim='member', skipna=True)
            ensemble_mean = ensemble_mean.expand_dims({'member':['ensemble_mean']})
            ds_all_members=xr.concat([ensemble_mean, ds_all_members], dim='member')

            output_file = f'files_yearly/yield_annual_relative_{experiment}_{crop_name}.nc'
            write_to_netcdf(ds_all_members, output_file)
            print(colored(f"[OK] Saved {output_file}", 'green'))
            # print(colored(f"Data saved to {output_file}", 'green'))
            ds_all.close()
        else:
            print(colored(f'[SKIP] No data for {experiment} - {crop_name}', 'red'))
        
