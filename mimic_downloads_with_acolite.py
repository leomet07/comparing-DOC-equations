import rasterio
import numpy as np
import os
import pandas as pd
import ee
import io
import requests
import multiprocessing
import sys
import datetime
from pprint import pprint
import matplotlib.pyplot as plt
from pprint import pprint
from rasterio.warp import calculate_default_transform, reproject, Resampling


sys.path.append("/home/leo/Documents/nasa/waterquality/comparing-DOC-equations/acolite")

# CLOUD_FILTER = 50

project = "leomet07-waterquality"


def open_gee_project(project: str):
    print(project)
    # try:
    #     ee.Initialize(opt_url="https://earthengine-highvolume.googleapis.com")
    # except Exception as e:
    ee.Authenticate()
    ee.Initialize(
        project=project, opt_url="https://earthengine-highvolume.googleapis.com"
    )


def import_assets(objectid: int, projectName: str) -> ee.FeatureCollection:
    LakeShp = ee.FeatureCollection(
        f"projects/{projectName}/assets/195-ALTM-ALAP-lakes-withCentroid"
    )
    LakeShp = ee.FeatureCollection(
        LakeShp.filter(
            ee.Filter.eq("OBJECTID", objectid)
        )  # use SITE_ID or name to get objectid
    )
    return LakeShp


open_gee_project(project=project)

import acolite.gee

open_gee_project(project=project)

target_dir = "all_lake_images_three_front_and_back_water_mask_level_2"

subfolders = list(os.listdir(target_dir))
subfolders.sort()
for subfolder in subfolders:
    tif_folder_path = os.path.join(target_dir, subfolder)
    if os.path.isfile(tif_folder_path):
        continue  # this is the log file
    if subfolder == "rondaxe,_lake_tifs" or subfolder == "otter_lake_tifs":
        continue  # temporary, rondaxe does not have enough pixels around centroid

    for filename in os.listdir(tif_folder_path):
        tif_filepath = os.path.join(tif_folder_path, filename)

        with rasterio.open(tif_filepath) as src:
            tags = src.tags()
            date_str = tags["date"]
            date = pd.to_datetime(date_str)  # satellite date
            objectid = int(float(tags["objectid"]))
            start_date = date - pd.DateOffset(days=1)
            end_date = date + pd.DateOffset(days=1)

            LakeShp = import_assets(objectid, project)  # get shape of lake

        # @markdown Basic settings
        isodate_start = str(start_date)
        isodate_end = str(end_date)
        print("isodate_start: ", isodate_start)
        sensors = "L8_OLI"  # @param ["L8_OLI"]
        output_dir = f"/home/leo/Documents/nasa/waterquality/comparing-DOC-equations/acolite_out/lake{objectid}_{date_str}/"  # @param {type:"string"}

        settings = {}
        gee_settings = {}

        gee_settings["output"] = output_dir
        gee_settings["sensors"] = sensors
        gee_settings["isodate_start"] = isodate_start
        gee_settings["isodate_end"] = isodate_end
        gee_settings["strict_subset"] = True
        gee_settings["run_hybrid_dsf"] = False
        gee_settings["run_offline_dsf"] = True

        # @markdown L2W parameters
        settings["l2w_parameters"] = []
        Rrs = True  # @param {type:"boolean"}
        if Rrs:
            settings["l2w_parameters"].append("Rrs_*")

        rhorc = True  # @param {type:"boolean"}
        if rhorc:
            settings["l2w_parameters"].append("rhorc_*")

        custom_l2w_parameters = ""  # @param {type:"string"}
        if custom_l2w_parameters:
            settings["l2w_parameters"] += custom_l2w_parameters.split(",")

        settings["l1r_export_geotiff"] = True
        settings["l2t_export_geotiff"] = True
        settings["l2r_export_geotiff"] = True
        settings["l2w_export_geotiff"] = True
        settings["output_geotiff"] = True

        coordinates = LakeShp.geometry().bounds().coordinates().getInfo()[0]
        lon = [x[0] for x in coordinates]
        lat = [x[1] for x in coordinates]
        S, W, N, E = min(lat), min(lon), max(lat), max(lon)
        gee_settings["limit"] = [S, W, N, E]

        acolite.gee.agh_run(settings=gee_settings, acolite_settings=settings)

        print("FINISHED DOWNLOADING FOR: ", date)

        # Now combine into one

        # find all output files
        output_files = list(
            filter(lambda x: "L2W" in x and "crop_Rrs" in x, os.listdir(output_dir))
        )

        base_stub = output_files[0]

        band_names = ["443", "483", "561", "655", "865"]

        output_bands = list(map(lambda x: [], band_names))
        for index in range(len(band_names)):
            band_name = band_names[index]

            input_filename = base_stub[:-7] + band_name + ".tif"

            input_file = os.path.join(output_dir, input_filename)

            new_crs = "EPSG:4326"
            # reproject raster to project crs
            with rasterio.open(input_file) as src:
                src_crs = src.crs
                transform, width, height = calculate_default_transform(
                    src_crs, new_crs, src.width, src.height, *src.bounds
                )
                kwargs = src.meta.copy()

                kwargs.update(
                    {
                        "crs": new_crs,
                        "transform": transform,
                        "width": width,
                        "height": height,
                    }
                )

                output_bands[index] = np.zeros(
                    (height, width), np.float64
                )  # create staging grounds to dump in later

                reproject(
                    source=rasterio.band(
                        src, 1
                    ),  # each band file for some reason has 1 real band and 2 fake bands written to it, we only care abt the first
                    destination=output_bands[index],
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=new_crs,
                    resampling=Resampling.nearest,
                )
        kwargs.update(count=len(band_names))

        true_output_dir = os.path.join(
            "/home/leo/Documents/nasa/waterquality/comparing-DOC-equations/all_acolite_true_out",
            subfolder,
        )
        if not os.path.exists(true_output_dir):
            os.makedirs(true_output_dir)

        output_file = os.path.join(true_output_dir, f"{subfolder}_{date_str}_ALL.tif")
        with rasterio.open(output_file, "w", **kwargs) as dst:
            dst.update_tags(**tags)

            for i in range(len(output_bands)):
                dst.write(output_bands[i], i + 1)

        print("\n\nSUCCESSFULLY WROTE AND COMBINED: ", output_file, "\n\n")
