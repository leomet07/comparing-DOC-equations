import os
import geopandas
import pandas as pd
import math
from tqdm import tqdm
from pprint import pprint
import rasterio
import fetch_landsat
import raster_utils

PROJECT = "leomet07-waterquality"

fetch_landsat.open_gee_project(project=PROJECT)

shapefile_path = os.path.join("doc-data", "195-ALTM-ALAP-lakes-withCentroid.shp")

shp_df = geopandas.read_file(shapefile_path)

# print(shp_df)
# print(shp_df.columns)

shp_df = shp_df[
    [
        "OBJECTID",
        "NAME",
        "FTYPE",
        "FCODE",
        "FCODE_DESC",
        "SQKM",
        "SQMI",
        "Permanent_",
        "Resolution",
        "GNIS_ID",
        "GNIS_Name",
        "AreaSqKm",
        "Elevation",
        "ReachCode",
        "FType_2",
        "Shape_Area",
        "NHDPlusID",
        "area_ha",
        "Lon-Cent",
        "Lat-Cent",
    ]
]


shp_df.to_csv("195_lake_shapefile.csv")

# Site information

site_information = pd.read_excel(os.path.join("ALTM", "Site_Information_2022_8_1.xlsx"))

# filter so that program_id = LTM_ALTM
site_information = site_information[site_information["PROGRAM_ID"] == "LTM_ALTM"]

# print(site_information)
# print(site_information.columns)

num_matches = 0
# now match shp file centroid to site information
for site_index, site_row in site_information.iterrows():  # O(n^2)
    acceptable_diff = 0.007

    for shp_index, shp_row in shp_df.iterrows():
        p1 = (site_row["LATDD_CENTROID"], site_row["LONDD_CENTROID"])  # lat, long
        p2 = (shp_row["Lat-Cent"], shp_row["Lon-Cent"])  # lat, long

        if math.dist(p1, p2) < acceptable_diff:
            site_id = site_row["SITE_ID"]

            print(
                f"{site_id}: Match between: SHP({shp_row["NAME"]}) and LTM_DATA({site_row["SITE_NAME"]}) | SHP_INDEX({shp_index}) | SHP_OBJECTID({shp_row["OBJECTID"]})"
            )
            num_matches += 1

            # shp_row["SITE_ID"] = site_id

            shp_df.at[shp_index, "SITE_ID"] = site_id  # modifies the original

print("# of matches: ", num_matches)

"""
# ---------------------- Identify some good testing lakes ------------------------------------
# Testing Lakes

print()

lake_names_of_interest = [  # these match SHP file
    "Woods Lake",
    "Big Moose Lake",
    "Brook Trout Lake",
    "Dart Lake",
    "G Lake",
    "Indian Lake",  # Indian Lake is outlier, row 12 was the actual match, so drop rows with no siteid
    "Squaw Lake",
    "Moss Lake",
    "Otter Lake",
    "Queer Lake",
    "Raquette Lake Reservoir",
    "Sagamore Lake",
    "Cascade Lake",
    "Limekiln Lake",
    "North Lake",
    "Rondaxe, Lake",
    "South Lake",  # missing matched site_id
]

valid_tuples_of_interest = []

for name_of_interest in lake_names_of_interest:
    matched_lakes_in_joined_shp_file = shp_df[shp_df["NAME"] == name_of_interest]

    matched_lakes_in_joined_shp_file = matched_lakes_in_joined_shp_file.dropna(
        subset=["SITE_ID"]
    )
    if len(matched_lakes_in_joined_shp_file) > 1:
        raise Exception("Multiple lakes in shapefile found with interested name")

    if len(matched_lakes_in_joined_shp_file) == 0:
        print(f'Lake of interest with name "{name_of_interest}" not found. ')
        continue

    valid_tuples_of_interest.append(
        (matched_lakes_in_joined_shp_file.iloc[0]["SITE_ID"], name_of_interest)
    )

print(valid_tuples_of_interest)

print()
"""
# -----------------------------------------------------------------------------------


truth_data = pd.read_excel(os.path.join("ALTM", "LTM_Data_2023_3_9.xlsx"))

truth_data = truth_data[
    ["SITE_ID", "DATE_SMP", "DOC_MG_L"]
]  # don't care abt other columns

# filter for after landsat 8 launch

truth_data = truth_data[truth_data["DATE_SMP"] > "2013-02-11"]

# Merge in shp file (object ids, centroids) into truth_data
truth_data = truth_data.merge(shp_df, left_on="SITE_ID", right_on="SITE_ID")

print("Truth data: \n", truth_data)

"""
# ------------------------- WOODS LAKE TESTING --------------------------------------
OUT_DIR = "woods_lake_tifs"
LAKEOBJECTID = 298284
LAKESITEID = "040576"

data_for_woods_lake = truth_data[truth_data["SITE_ID"] == LAKESITEID]  # or use OBJECTID

print(data_for_woods_lake)

dates_for_woods_lake = (
    data_for_woods_lake["DATE_SMP"].sort_values().tolist()
)  # ascending order
# to check for flyovers in satellite download library

# -----------------------------------------------------------------------------------

# Check for sateliite flyover dates for woods lake at these dates:

print("Dates to check for woods lake: ", dates_for_woods_lake)

successful_tifs = []  # (date, filepath)
if not os.path.exists(OUT_DIR):  # quick cache-ing
    for i in tqdm(range(len(dates_for_woods_lake))):
        start_date = dates_for_woods_lake[i]
        end_date = start_date + pd.DateOffset(
            days=1
        )  # check for 1, 3, 5 days (this is one day)

        start_date_YYYY_MM_DD = str(start_date)[:10]  # faster than strftime
        print("Searching for images on: ", start_date_YYYY_MM_DD)

        try:
            out_tif_filepath = fetch_landsat.export_raster_main_landsat(
                out_dir=OUT_DIR,
                out_filename=f"woods_lake_{start_date_YYYY_MM_DD}.tif",  # woods lake
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
else:
    successful_tifs = list(
        map(lambda filename: os.path.join(OUT_DIR, filename), os.listdir(OUT_DIR))
    )

pprint(successful_tifs)


# ----------------------------------------------
# Fetch at centroid

for file in successful_tifs:
    lake_info = shp_df[shp_df["OBJECTID"] == LAKEOBJECTID]
    lat = lake_info["Lat-Cent"]
    long = lake_info["Lon-Cent"]
    circle_at_centroid = raster_utils.get_circular_section_from_file(
        file, lat, long, 60
    )  # 60m radius

    pprint(circle_at_centroid)
    # now print mean for every band
    print("\n\n")
"""
