# Code originally from https://github.com/leomet07/PredictChlorophyllALibrary

import ee
import os
import pandas as pd
import io
import requests
import rasterio
from rasterio.transform import from_bounds
import multiprocessing
import sys
import datetime
from pprint import pprint
import matplotlib.pyplot as plt

## GLOBAL CONSTANTS FOR THIS PROJECT
CLOUD_FILTER = 50


def visualize(tif_path: str):
    # Open the GeoTIFF file
    with rasterio.open(tif_path) as src:
        # Read the number of bands and the dimensions
        num_bands = src.count
        height = src.height
        width = src.width
        tags = src.tags()
        title = f"Date: {tags["date"]}, GNIS_ID: {tags["GNIS_ID"]}, Scale: {tags["scale"]}\n"

        print(f"Number of bands: {num_bands}")
        # print(f"Dimensions: {width} x {height}")

        # Read the entire image into a numpy array (bands, height, width)
        img = src.read()
        # Display each band separately
        fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(15, 10))

        for i, ax in enumerate(axes.flatten()):
            if i < num_bands:
                ax.imshow(img[i, :, :], cmap="gray")  # Display each band separately
                ax.set_title(f"Band {i+1}")
                ax.axis("off")
                # print(img[i, :, :])
        plt.tight_layout()
        plt.suptitle(title, fontsize=24)  # Super title for all the subplots!
        plt.show()


def see_if_all_image_bands_valid(band_values):  # min values passed in here usually
    # for band in band_values:
    #     print(
    #         "band: ", band, band_values[band]
    #     )  # catches if band is empty for some reason
    for band in band_values:
        if band_values[band] != None:
            return True
    # if it made it all the way here, all values in this dict are None
    return False


"""
Set up GEE account and get the name of your GEE project.
"""


def open_gee_project(project: str):
    try:
        ee.Initialize(opt_url="https://earthengine-highvolume.googleapis.com")
    except Exception as e:
        ee.Authenticate()
        ee.Initialize(
            project=project, opt_url="https://earthengine-highvolume.googleapis.com"
        )


"""
Import assets from gee depending on which lake you want an image of
"""


def import_assets(GNIS_ID: str, projectName: str) -> ee.FeatureCollection:
    LakeShp = ee.FeatureCollection(
        f"projects/{projectName}/assets/adk-lakes-gt25km2-200m-bound-clean"
    )
    LakeShp = ee.FeatureCollection(
        LakeShp.filter(
            ee.Filter.eq("GNIS_ID", GNIS_ID)
        )  # use SITE_ID or name to get objectid
    )
    return LakeShp


"""## Creating mask functions

These functions mask clouds based on the QA_PIXEL band (maskL8sr), select pixels that are >= 75% water (jrcMask), and a 30m buffer around roads to mask bridges (roadMask)
"""


# Creating mask functions
# function to mask out clouds
def maskL5sr(image):
    # Bits 2, 3, and 4, are cirrus, cloud  and cloudshadow, respectively.
    cloudsBitMask = 1 << 3
    cloudShadowBitMask = 1 << 4
    # cirrusBitMask = 1 << 2
    waterBitMask = 1 << 7  # 1 means water

    # Get the pixel QA band.
    qa = image.select("QA_PIXEL")
    # Both flags should be set to zero, indicating clear conditions.
    mask = (
        qa.bitwiseAnd(cloudShadowBitMask)
        .eq(0)
        .And(qa.bitwiseAnd(cloudsBitMask).eq(0))  # want not high cloud confidence
        # .And(qa.bitwiseAnd(cirrusBitMask).eq(0))
        .And(qa.bitwiseAnd(waterBitMask).neq(0))  # keep water
        # tho question, if 1 is water, then why is this working if we are keeping it when it is zero(land)
    )
    return image.updateMask(mask)


def import_collections(filter_range, LakeShp) -> ee.Image:
    """## Buffer function for points"""
    # # filter landsat 8 and 9 scenes by path / row
    FC_OLI = (
        ee.ImageCollection("LANDSAT/LT05/C02/T1_L2")  # level 1 (T1_L2 would be level 2)
        .filterMetadata("CLOUD_COVER", "less_than", CLOUD_FILTER)
        .filter(filter_range)
        .filterBounds(LakeShp)
        .map(maskL5sr)
        .sort("system:time_start")
    )

    FC_OLI = FC_OLI.select(["SR_B1", "SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B7"])

    return FC_OLI


def get_image_and_date_from_image_collection(coll, index, shp):
    image = ee.Image(coll.toList(coll.size()).get(index))
    image_index = image.get("system:index").getInfo()
    date = ee.Date(image.get("system:time_start")).format("YYYY-MM-dd").getInfo()
    image = (
        image.select("SR_B.").multiply(0.0000275).add(-0.2)
    )  # scale back BEFORE clipping
    image = image.clip(shp)
    image = image.toFloat()

    return image, image_index, date


def get_raster(start_date, end_date, LakeShp, scale) -> ee.Image:
    date_range = ee.Filter.date(start_date, end_date)
    filter_range = ee.Filter.Or(date_range)

    merged_landsat_image_collection = import_collections(filter_range, LakeShp)
    merged_landsat_image_collection_len = (
        merged_landsat_image_collection.size().getInfo()
    )

    if merged_landsat_image_collection_len == 0:
        raise Exception("NO IMAGES FOUND")

    for i in range(0, merged_landsat_image_collection_len):
        image, image_index, date = get_image_and_date_from_image_collection(
            merged_landsat_image_collection, i, LakeShp
        )

        min_value = image.reduceRegion(
            reducer=ee.Reducer.min(),
            geometry=LakeShp.geometry(),  # or your specific geometry
            scale=scale,
            maxPixels=1e9,
            crs="EPSG:4326",
        ).getInfo()

        if see_if_all_image_bands_valid(min_value):
            return image, image_index, date
    # if it made it here, all have blank images (due to NASA JPL aggressive cloud alterer/filter)
    raise Exception("IMAGE IS ALL BLANK :(((")


def export_raster_main_landsat_L2(
    out_dir: str,
    out_filename: str,
    project: str,
    lakeid: str,
    start_date: str,
    end_date: str,
    insitu_date: str,
    scale: int,
    shouldVisualize: bool = False,
    index_logfile_path=None,
):
    LakeShp = import_assets(lakeid, project)  # get shape of lake
    image, image_index, date = get_raster(
        start_date=start_date, end_date=end_date, LakeShp=LakeShp, scale=scale
    )

    url = image.getDownloadURL(
        {
            "format": "GEO_TIFF",
            "scale": scale,  #  increasing this makes predictions more blocky but reduces request size (smaller means more resolution tho!)
            "region": LakeShp.geometry(),
            "filePerBand": False,
            "crs": "EPSG:4326",
        }
    )

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # export!
    out_filepath = os.path.join(out_dir, out_filename)
    # download image, and then view metadata with rasterio
    # print("Downloading raster...")

    response = requests.get(url)
    with open(out_filepath, "wb") as f:
        f.write(response.content)

    new_metadata = {
        "date": date,
        "closest_insitu_date": insitu_date,  # this was the date from the insitu
        "GNIS_ID": lakeid,
        "scale": scale,
        "satellite": "landsat5",
        "image_index": image_index,
        "algorithm": "L2",
    }
    with rasterio.open(out_filepath, "r+") as dst:
        dst.update_tags(**new_metadata)

    if shouldVisualize:
        print(f"Image saved to {out_filepath}")
        print("Saved image metadata: ", new_metadata)
        visualize(out_filepath)

    if index_logfile_path:
        with open(index_logfile_path, "a") as index_logfile:
            index_logfile.write(f"{image_index}\n")

    return out_filepath


if __name__ == "__main__":
    if len(sys.argv) != 8:
        print(
            "python fetch_landsat_five_L2.py <out_dir> <project> <lakeid> <start_date> <end_date> <scale> <out_filename>"
        )
        sys.exit(1)

    out_dir = sys.argv[1]
    project = sys.argv[2]
    lakeid = sys.argv[3]
    start_date = sys.argv[4]  # STR, in format YYYY-MM-DD
    end_date = sys.argv[5]  # STR, in format YYYY-MM-DD
    scale = int(sys.argv[6])
    out_filename = sys.argv[7]

    open_gee_project(project=project)

    export_raster_main_landsat_L2(
        out_dir=out_dir,
        out_filename=out_filename,
        project=project,
        lakeid=lakeid,
        start_date=start_date,
        end_date=end_date,
        insitu_date=start_date,  # this doesnt matter when only fetching one, just for testing
        scale=scale,
        shouldVisualize=True,
    )
