from tqdm import tqdm
import inspect_shapefile
import os
import pandas as pd
import fetch_landsat
from pprint import pprint

PROJECT = "leomet07-waterquality"

fetch_landsat.open_gee_project(project=PROJECT)

LAKEOBJECTID = 298315

lake_name = inspect_shapefile.shp_df[
    inspect_shapefile.shp_df["OBJECTID"] == LAKEOBJECTID
]["NAME"].item()
lake_name = lake_name.lower().replace(" ", "_")
print("Lake name: ", lake_name)

OUT_DIR = f"{lake_name}_tifs"

data_for_lake = inspect_shapefile.truth_data[
    inspect_shapefile.truth_data["OBJECTID"] == float(LAKEOBJECTID)
]

print(data_for_lake)

dates_for_lake = data_for_lake["DATE_SMP"].sort_values().tolist()  # ascending order

# Check for sateliite flyover dates for woods lake at these dates:
print("Dates to check for lake: ", dates_for_lake)

successful_tifs = []  # (date, filepath)
for i in tqdm(range(len(dates_for_lake))):
    start_date = dates_for_lake[i]
    end_date = start_date + pd.DateOffset(
        days=5
    )  # check for 1, 3, 5 days (this is one day)

    start_date_YYYY_MM_DD = str(start_date)[:10]  # faster than strftime
    print("Searching for images on: ", start_date_YYYY_MM_DD)

    try:
        out_tif_filepath = fetch_landsat.export_raster_main_landsat(
            out_dir=OUT_DIR,
            out_filename=f"{lake_name}_{start_date_YYYY_MM_DD}.tif",  # woods lake
            project=PROJECT,
            lakeid=LAKEOBJECTID,
            start_date=start_date,
            end_date=end_date,
            scale=30,
            shouldVisualize=False,
        )
        successful_tifs.append(out_tif_filepath)
    except Exception as e:
        print(e)

pprint(successful_tifs)
