#!/usr/bin/env python3

import sys
from lib.crop import crop_northern_hemisphere


def main():
    if len(sys.argv) < 3:
        print("Usage:")
        print("python scripts/crop_nh.py <input.tif> <output.tif> [compression]")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    # Optional compression argument
    compress = sys.argv[3] if len(sys.argv) > 3 else "LZW"

    crop_northern_hemisphere(
        in_file=input_file,
        out_file=output_file,
        compress=compress
    )

    print("Cropping completed successfully.")


if __name__ == "__main__":
    main()