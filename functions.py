import xarray as xr
import intake
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import pprint
import numpy as np

def get_data_one_model(cat_url, variable_id,
    experiment_id = ["piControl"],
    table_id = "Amon",
    member_id = "r1i1p1f1",
    grid_label = "gn"):
    """
    Getting the data from CMIP
    """
    col = intake.open_esm_datastore(cat_url)
    zstore = "gs://cmip6/CMIP6/CMIP/HAMMOZ-Consortium/MPI-ESM-1-2-HAM/piControl/r1i1p1f1/Amon/clt/gn/v20190627/"
    cat = col.search(experiment_id = experiment_id,
                    variable_id = variable_id,
                    table_id = table_id,
                    member_id = member_id,
                    grid_label = grid_label,
                    zstore = zstore)
    dset_dict = cat.to_dataset_dict(zarr_kwargs={'use_cftime':True})
    return dset_dict

def time_mean_func(data_array):
    time_mean = data_array.mean("time")
    return time_mean

def area_mean_func(data_array):
    weights = np.cos(np.deg2rad(data_array.lat)) #Creating weights
    weights.name = "weights"
    weighted_array = data_array.weighted(weights)
    area_mean = weighted_array.mean(("lon", "lat"))
    return area_mean

def means(data_array):
    time_mean = time_mean_func(data_array)
    area_and_time_mean = area_mean_func(data_array)
    return area_and_time_mean

def plot_arctic(data_array, variable, filename, one_layer = False, time_slice = 0, savefig = False):
    """
    Plots everything in the actic
    """
    #Plotting of the dataset
    fig = plt.figure(1, figsize=[10,10])

    ax = plt.subplot(1, 1, 1, projection=ccrs.Orthographic(0, 90))
    ax.coastlines()

    # Fix extent
    minval = 0
    maxval = 100

    # pass extent with vmin and vmax parameters
    if one_layer == True:
        data_array[variable].isel(time=time_slice).plot(ax=ax, vmin=minval, vmax=maxval, transform=ccrs.PlateCarree(), cmap='Blues')
    else:
        data_array[variable].plot(ax=ax, vmin=minval, vmax=maxval, transform=ccrs.PlateCarree(), cmap='Blues')
    if savefig == True:
        plt.savefig(filename)
    plt.show()

def plot_graph(x, y, filename, savefig = False):
    plt.plot(x, y)
    if savefig == True:
        plt.savefig(filename)
    plt.show()

def slice(data_array, time = True, area = False):
    """
    Slices the data for time and area
    """
    if time == True:
        # Do the time slice
        starttime = "1850-01-16"
        endtime = "1880-01-16"
        data_array = data_array.sel(time = slice(starttime, endtime))
    if area == True:
        # Do the area slice, longdetudes
        """
        Antar dette er riktig, at latitude 60 - 90 er over arktis
        """
        minlat = 60
        maxlat = 90
        data_array = data_array.sel(lat = slice(minlat, maxlat))
    return data_array

cat_url = "https://storage.googleapis.com/cmip6/pangeo-cmip6.json"
dataset_dict_clt = get_data_one_model(cat_url, ["clt"])
# dataset_dict_tas = get_data_one_model(cat_url, ["tas"])
original_dataset_dict = dataset_dict_clt

for key in dataset_dict_clt:
    data_set = dataset_dict_clt[key]
    print(f"Data set before slicing: | {data_set}")
    data_set = slice(data_set, area = False)
    print(f"Data set after slicing: | {data_set}")
    # data_set_means = means(data_set)
    # print(f"Data set after taking the means: | {data_set}")
