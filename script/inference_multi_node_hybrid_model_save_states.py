#!/usr/bin/env python3

import sys
#from lib.inference_lstm_2 import run_inference
#import lib.inference_lstm_2
#from lib.inference_lstm import run_inference_multi_node_hybrid_model

from lib.inference_lstm import run_inference_multi_node_hybrid_model_with_saving_states


def main():
#    if len(sys.argv) < 3:
#        print("Usage:")
#        print("python scripts/interpolate_missing_day.py <missing_file.tif> <output_dir>")
#        sys.exit(1)
#
#    missing_file = sys.argv[1]
#    output_dir = sys.argv[2]

#    print("inside script run_inference_multi_node_hybrid_model")
    print("inside script run_inference_multi_node_hybrid_model_with_saving_states")

    try:
#        run_inference_multi_node_hybrid_model()
        run_inference_multi_node_hybrid_model_with_saving_states()
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)

    print("inference completed successfully.")


if __name__ == "__main__":
    main()