#!/usr/bin/env python3

import sys
from lib.interpolate import interpolate_missing_day


def main():
    if len(sys.argv) < 3:
        print("Usage:")
        print("python scripts/interpolate_missing_day.py <missing_file.tif> <output_dir>")
        sys.exit(1)

    missing_file = sys.argv[1]
    output_dir = sys.argv[2]

    try:
        interpolate_missing_day(
            missing_file_path=missing_file,
            output_dir=output_dir
        )
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    print("Interpolation completed successfully.")


if __name__ == "__main__":
    main()