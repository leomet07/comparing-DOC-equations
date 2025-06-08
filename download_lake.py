from tqdm import tqdm
import inspect_shapefile
import os
import pandas as pd
import fetch_landsat
from pprint import pprint
import multiprocessing
import random

PROJECT = "leomet07-waterquality"

fetch_landsat.open_gee_project(project=PROJECT)


def gen_all_lakes_all_dates_params(OUT_DIR):
    all_params = []
    for lake_info in inspect_shapefile.lake_infos_of_interest:
        lake_name = lake_info["NAME"].lower().replace(" ", "_")
        lake_objectid = lake_info["OBJECTID"]
        lake_out_dirname = f"{lake_name}_tifs"
        lake_out_dir_path = os.path.join(OUT_DIR, lake_out_dirname)

        data_for_lake = inspect_shapefile.truth_data[
            inspect_shapefile.truth_data["OBJECTID"] == float(lake_objectid)
        ]

        dates_for_lake = (
            data_for_lake["DATE_SMP"].sort_values().tolist()
        )  # ascending order

        # Check for sateliite flyover dates for woods lake at dates
        for i in range(len(dates_for_lake)):
            start_date = dates_for_lake[i]
            end_date = start_date + pd.DateOffset(
                days=5
            )  # check for 1, 3, 5 days (this is one day)

            start_date_YYYY_MM_DD = str(start_date)[:10]  # faster than strftime

            out_filename = f"{lake_name}_{start_date_YYYY_MM_DD}.tif"

            all_params.append(
                (
                    lake_out_dir_path,
                    out_filename,
                    PROJECT,
                    lake_objectid,
                    start_date,
                    end_date,
                    30,  # scale
                    False,  # Should visualize
                )
            )

    return all_params


def wrapper_export(
    args,
):  # this function allows ONE param to be spread onto many params for a function
    fetch_landsat.export_raster_main_landsat(*args)


if __name__ == "__main__":
    manager = multiprocessing.Manager()
    scale_cache = manager.dict()  # empty by default
    pool = multiprocessing.Pool(25)

    all_params_to_pass_in = gen_all_lakes_all_dates_params("TEST")

    random.shuffle(all_params_to_pass_in)

    # Starmap
    pool.imap(
        wrapper_export,
        tqdm(all_params_to_pass_in, total=len(all_params_to_pass_in)),
    )
    pool.close()
    pool.join()
