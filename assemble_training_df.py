import tqdm
import rasterio
import numpy as np
import pandas as pd
import os
from matplotlib import pyplot as plt
from shapely.geometry import Point
import rasterio.features
import sys

if len(sys.argv) <= 1:
    print(
        "You need to specify which algorithm out folders you are assembling as arguments."
    )
    sys.exit(1)

import inspect_shapefile


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

        # replace all -infs with nan
        bands[~np.isfinite(bands)] = np.nan

        if (
            "L2" in tif_path
        ):  # level 2 data correction https://www.usgs.gov/landsat-missions/landsat-collection-2-surface-reflectance
            bands = bands * 0.0000275 - 0.2  # operations on nan/inf are still nan/inf

        bands[bands > 0.1] = (
            np.nan
        )  # keep array shape but remove high reflectance outliers (clouds)
        # also this removes acolite outofbound nan values (10^36)

        return (
            bands,
            profile,
            transform,
            scale,
            x_res,
            closest_insitu_date,
            objectid,
        )


def add_training_entries_from_algorithim_out_folder(out_folder, training_entries):
    algorithim_name = out_folder.split("_")[-1].upper().replace("/", "")
    subfolders = list(os.listdir(out_folder))
    subfolders.sort()
    for subfolder in subfolders:
        if os.path.isfile(os.path.join(out_folder, subfolder)):
            continue  # this is the log file
        if subfolder == "rondaxe,_lake_tifs" or subfolder == "otter_lake_tifs":
            continue  # temporary, rondaxe does not have enough pixels around centroid

        tif_folder_path = os.path.join(out_folder, subfolder)

        for filename in os.listdir(tif_folder_path):
            current_training_entry = {}
            tif_filepath = os.path.join(tif_folder_path, filename)

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
            except rasterio.errors.RasterioIOError:
                continue

            # matched doc
            all_chla = inspect_shapefile.truth_data[
                (inspect_shapefile.truth_data["OBJECTID"] == float(objectid))
                & (inspect_shapefile.truth_data["DATE_SMP"] == closest_insitu_date)
            ]["CHL_A_UG_L"]
            try:
                chla = all_chla.item()
            except ValueError:  # array either has 2+ or 0 items
                if len(all_chla) > 0:  # means 2+ measurements for that date, take mean
                    chla = all_chla.mean()
                else:
                    raise Exception("No CHLA values found for that date.")
            current_training_entry["chla"] = chla
            # get lat and long
            centroid_lat = inspect_shapefile.truth_data[
                (inspect_shapefile.truth_data["OBJECTID"] == float(objectid))
                & (inspect_shapefile.truth_data["DATE_SMP"] == closest_insitu_date)
            ]["Lat-Cent"].iloc[
                0
            ]  # take first entry, lake centroid lat will be the same for any matched insitu
            centroid_long = inspect_shapefile.truth_data[
                (inspect_shapefile.truth_data["OBJECTID"] == float(objectid))
                & (inspect_shapefile.truth_data["DATE_SMP"] == closest_insitu_date)
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

            not_enough_pixels = False
            for band in bands:
                valid_pixels = band[np.isfinite(band)]
                if len(valid_pixels) < 3:
                    not_enough_pixels = True

            if not_enough_pixels:
                continue

            # now, can take means safely
            band_names = ["443", "483", "561", "655", "865"]
            for band_index in range(
                len(band_names)
            ):  # in case tif has extra bands we don't care about
                band = bands[band_index]
                band_name = band_names[band_index]
                mean_value = np.nanmean(band)
                current_training_entry[band_name] = mean_value

            # Recod alg name
            current_training_entry["alg"] = algorithim_name
            current_training_entry["lakeid"] = objectid

            training_entries.append(current_training_entry)


training_entries = []


for folder in sys.argv[1:]:
    add_training_entries_from_algorithim_out_folder(folder, training_entries)


training_df = pd.DataFrame(training_entries)
print("Training df: \n", training_df)

# first get just the dir name (handles forward slashes) then get basename
stub = os.path.basename(os.path.dirname(sys.argv[1]))
csv_filepath = f"{stub}_training_data.csv"

training_df.to_csv(csv_filepath, index=False)
