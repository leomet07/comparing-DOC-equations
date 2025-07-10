from tqdm import tqdm
import inspect_shapefile
import os
import pandas as pd
import fetch_landsat
from pprint import pprint
import multiprocessing
import random
import sys


def gen_all_lakes_all_dates_params(project, OUT_DIR, start_date_range, end_date_range):
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

        date_range = list(
            pd.date_range(
                start=start_date_range,
                end=end_date_range,
                freq=f"8D",  # together, landsat 8/9 revisit every scene every 8d
            )
        )

        for i in range(len(date_range) - 1):  # stop iterating one element short
            start_date = date_range[i].strftime(f"%Y-%m-%d")
            end_date = date_range[i + 1].strftime(f"%Y-%m-%d")
            # check for 1, 3, 5 days (this is one day)

            out_filename = f"{lake_name}_{start_date}.tif"

            all_params.append(
                (
                    lake_out_dir_path,
                    out_filename,
                    project,
                    lake_objectid,
                    start_date,
                    end_date,
                    "NOT_FETCHING_BY_INSITU_DATE",  # so this is known in image metadata
                    30,  # scale
                    False,  # Should visualize
                    index_logfile_path,  # logfile path for image indexes
                )
            )

    return all_params


def wrapper_export(
    args,
):  # this function allows ONE param to be spread onto many params for a function
    fetch_landsat.export_raster_main_landsat(*args)


if __name__ == "__main__":
    project = sys.argv[1]
    out_dir = sys.argv[2]

    fetch_landsat.open_gee_project(project=project)

    manager = multiprocessing.Manager()
    scale_cache = manager.dict()  # empty by default
    pool = multiprocessing.Pool(25)

    all_params_to_pass_in = gen_all_lakes_all_dates_params(
        project, out_dir, "2013-01-01", "2024-12-31"
    )

    random.shuffle(all_params_to_pass_in)

    # Starmap
    pool.imap(
        wrapper_export,
        tqdm(all_params_to_pass_in, total=len(all_params_to_pass_in)),
    )
    pool.close()
    pool.join()
