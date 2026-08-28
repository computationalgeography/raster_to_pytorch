#!/usr/bin/env python3

import sys
#from lib.inference_lstm_2 import run_inference
#import lib.inference_lstm_2
from lib.inference_lstm import run_inference_with_mask


def main():
#    if len(sys.argv) < 3:
#        print("Usage:")
#        print("python scripts/interpolate_missing_day.py <missing_file.tif> <output_dir>")
#        sys.exit(1)
#
#    missing_file = sys.argv[1]
#    output_dir = sys.argv[2]

    print("inside script run_inference_with_mask")

    try:
        run_inference_with_mask()
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)

    print("inference completed successfully.")


if __name__ == "__main__":
    main()