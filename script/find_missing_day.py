#!/usr/bin/env python3

import sys

from lib.find_missing_data import find_missing_dates


def main() -> None:
    if len(sys.argv) < 3:
        print("Usage:")
        print(
            "python scripts/find_missing_day.py <searching_folder_path> <output_path>"
        )
        sys.exit(1)

    searching_folder_path = sys.argv[1]
    output_path = sys.argv[2]

    try:
        find_missing_dates(searching_folder_path, output_path)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
