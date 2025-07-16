import tqdm
import rasterio
import numpy as np
import pandas as pd
import os
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import inspect_shapefile
from shapely.geometry import Point
from sklearn.metrics import r2_score, root_mean_squared_error
import rasterio.features
import warnings
from sklearn.linear_model import LinearRegression


def get_bands_from_tif(tif_path):
    with rasterio.open(tif_path) as src:
        profile = src.profile  # Get the profile of the existing raster
        transform = src.transform
        tags = src.tags()
        scale = tags["scale"]
        x_res = src.res[0]  # same as src.res[1]
        closest_insitu_date = tags["closest_insitu_date"]
        objectid = tags["objectid"]

        bands = src.read()

        # remove all infs currently in tif
        bands[~np.isfinite(bands)] = np.nan

        if "acolite" in tif_path:
            # weird large value as nan bug in acolite tifs
            bands[bands > 1] = (
                np.nan
            )  # surface reflactances should be < 1, remove the weird 9.96921e+36 values from acolite tifs

        if (
            "L2" in tif_path
        ):  # level 2 data correction https://www.usgs.gov/landsat-missions/landsat-collection-2-surface-reflectance
            bands = bands * 0.0000275 - 0.2  # operations on nan/inf are still nan/inf

        return (
            bands,
            profile,
            transform,
            scale,
            x_res,
            closest_insitu_date,
            objectid,
        )


def get_band_means(out_folder, subfolder, flyover_date):
    band_pixels_at_centroid_across_this_year = [[] for _ in range(5)]

    tif_folder_path = os.path.join(out_folder, subfolder)

    for filename in os.listdir(tif_folder_path):
        tif_filepath = os.path.join(tif_folder_path, filename)

        if flyover_date not in filename:  # initial july 2020 plots
            continue
        # filter filepath to a certian year only

        try:
            (
                bands,
                profile,
                transform,
                scale,
                x_res,
                closest_insitu_date,
                objectid,
            ) = get_bands_from_tif(tif_filepath)
        except rasterio.errors.RasterioIOError as e:
            continue

        # get lat and long
        centroid_lat = inspect_shapefile.truth_data[
            (inspect_shapefile.truth_data["OBJECTID"] == float(objectid))
        ]["Lat-Cent"].iloc[
            0
        ]  # take first entry, lake centroid lat will be the same for any matched insitu
        centroid_long = inspect_shapefile.truth_data[
            (inspect_shapefile.truth_data["OBJECTID"] == float(objectid))
        ]["Lon-Cent"].iloc[
            0
        ]  # take first entry, lake centroid long will be the same for any matched insitu

        radius_in_meters = 60
        circle = Point(centroid_long, centroid_lat).buffer(
            x_res * (radius_in_meters / float(scale))
        )  # however many x_res sized pixels needed for buffer of radius at downloaded scale

        outside_circle_mask = rasterio.features.geometry_mask(
            [circle], bands[0].shape, transform
        )

        for band in bands:
            band[outside_circle_mask] = (
                np.nan
            )  # arrays store pointer to ratio array, this is okay bc just a mutation

        # ------------------------------------------------------------
        # MAKE SURE THAT FOR THIS TIFF, CENTROID MEAN ACTUALLY EXISTS (consisting of at least 3 pixels)
        is_any_mean_ratio_nan = False
        list_of_input_tuples = []

        for band_index in range(len(bands)):
            band = bands[band_index]
            band_flatten = band.flatten()

            #  number of values to be averaging
            keep_valid_pixels_mask = np.isfinite(band_flatten)
            valid_pixels = band_flatten[keep_valid_pixels_mask]
            valid_pixels = valid_pixels[valid_pixels < 0.1]

            if len(valid_pixels) < 3:  # cloudy
                break

            band_pixels_at_centroid_across_this_year[band_index].extend(valid_pixels)

    band_means = []

    for i in range(5):
        # this will be warning if empty and will return nan, causing blank graph
        if len(band_pixels_at_centroid_across_this_year[i]) == 0:
            band_meawns = []  # reset, don't use this image
            break
        band_means.append(np.nanmean(band_pixels_at_centroid_across_this_year[i]))

    return band_means


flyover_date = "2021-08-"
sample_out_folder = (
    "all_flyover_of_lakes_main"  # just for getting the different lake subfolders
)
subfolders = list(os.listdir(sample_out_folder))
subfolders.sort()
for subfolder in subfolders:
    if os.path.isfile(os.path.join(sample_out_folder, subfolder)):
        continue  # this is the log file
    if (
        subfolder == "rondaxe,_lake_tifs"
        or subfolder == "otter_lake_tifs"
        or subfolder == "queer_lake_tifs"
    ):
        continue  # temporary, rondaxe does not have enough pixels around centroid

    band_names = [f"B{i+1}" for i in range(5)]

    band_means_l2 = get_band_means("all_flyover_of_lakes_L2", subfolder, flyover_date)

    band_means_main = get_band_means(
        "all_flyover_of_lakes_main", subfolder, flyover_date
    )
    band_means_acolite = get_band_means("all_flyover_acolite", subfolder, flyover_date)

    if len(band_means_l2) != 0:
        print("L2 plotted:", subfolder)
        plt.plot(band_names, band_means_l2, color="green")
    if len(band_means_main) != 0:
        print("MAIN plotted:", subfolder)
        plt.plot(band_names, band_means_main, color="blue")
    if len(band_means_acolite) != 0:
        print("ACOLITE plotted:", subfolder)
        try:
            plt.plot(band_names, band_means_acolite, color="red")
        except Exception as e:
            print(e)


plt.xlabel("Band")
plt.ylabel("Mean Reflectance")
plt.title(f"2021 August Mean Reflectances")
plt.legend(
    handles=[
        Line2D([0], [0], color="green", lw=4, label="L2"),
        Line2D([0], [0], color="blue", lw=4, label="MAIN"),
        Line2D([0], [0], color="red", lw=4, label="ACOLITE"),
    ]
)
plt.show()
