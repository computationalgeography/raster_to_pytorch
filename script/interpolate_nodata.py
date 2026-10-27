#!/usr/bin/env python3

import sys
from lib.interpolate_nodata_fill import nodata_fill_v2

def main():

    if len(sys.argv) != 5:
        print(
            "Usage:\n"
            "python nodata_fill.py <current.tif> <mask.tif> <output_folder> <partition_size>"
        )
        sys.exit(1)
        
    print("inside script interpolate_nodata_fill")

    current_file = sys.argv[1]
    mask_file = sys.argv[2]
    output_path = sys.argv[3]
    partition_size = int(sys.argv[4])

    nodata_fill_v2(current_file, mask_file, output_path, partition_size)


if __name__ == "__main__":
    main()