import sys
import os
from datetime import datetime, timedelta

from lib.interpolate_nodata_fill import nodata_fill     #nodata_fill_v2

def main():

    print("inside interpolate_nodate_test")

#    nodata_fill_v2(
    nodata_fill(
    '/archive/depfg/compgeo/projecto/data/pr/CHELSA_pr_15_07_2006_V.2.1.tif',
    '/archive/depfg/compgeo/projecto/data/inference_mask.tif',
    '/archive/depfg/compgeo/projecto/interpolate_nodata/pr/test.tif',
    1000)

 

if __name__ == "__main__":
    main()


