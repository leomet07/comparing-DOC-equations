from tqdm import tqdm
import inspect_shapefile
import os
import pandas as pd
from pprint import pprint
import multiprocessing
import random
import sys


def gen_all_lakes_all_dates_params(project, OUT_DIR, days_before_and_after_insitu: int):
    all_params = []

    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)

    index_logfile_path = os.path.join(OUT_DIR, "image_indexes_saved.csv")
    with open(index_logfile_path, "w") as index_logfile:
        index_logfile.write("image_index\n")

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
            start_date = dates_for_lake[i] - pd.DateOffset(
                days=days_before_and_after_insitu
            )
            end_date = dates_for_lake[i] + pd.DateOffset(
                days=days_before_and_after_insitu
            )  # check for 1, 3, 5 days (this is one day)

            start_date_YYYY_MM_DD = str(start_date)[:10]  # faster than strftime

            out_filename = f"{lake_name}_{start_date_YYYY_MM_DD}.tif"

            all_params.append(
                (
                    lake_out_dir_path,
                    out_filename,
                    project,
                    lake_objectid,
                    start_date,
                    end_date,
                    dates_for_lake[i],
                    30,  # scale
                    False,  # Should visualize
                    index_logfile_path,  # logfile path for image indexes
                )
            )

    return all_params


def wrapper_export(
    args,
):  # this function allows ONE param to be spread onto many params for a function
    export_raster_main_landsat(*args)


if __name__ == "__main__":
    project = sys.argv[1]
    out_dir = sys.argv[2]
    alg = str(sys.argv[3]).upper()

    if alg == "L2":
        from fetch_landsat_L2 import (
            export_raster_main_landsat_L2 as export_raster_main_landsat,
            open_gee_project,
        )
    elif alg == "MAIN":
        from fetch_landsat import export_raster_main_landsat, open_gee_project
    else:
        print(f'ALG "{alg}" is not supported.')

    print("Using ALG: ", alg)

    days_before_and_after_insitu = int(sys.argv[4])

    open_gee_project(project=project)

    manager = multiprocessing.Manager()
    scale_cache = manager.dict()  # empty by default
    pool = multiprocessing.Pool(25)

    all_params_to_pass_in = gen_all_lakes_all_dates_params(
        project, out_dir, days_before_and_after_insitu
    )

    random.shuffle(all_params_to_pass_in)

    # Starmap
    pool.imap(
        wrapper_export,
        tqdm(all_params_to_pass_in, total=len(all_params_to_pass_in)),
    )
    pool.close()
    pool.join()
