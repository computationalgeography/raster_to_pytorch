
#!/bin/bash

PROJECT_ROOT="/projects/prjs1930/saeb/20260218_Oriol_snow_model_LSTM"
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

#python script/inference.py

#python script/inference_with_mask.py

#python -m memory_profiler script/inference.py > memory_profile_inference.txt

python -m scalene --outfile profile_cpu_mem.txt script/inference_multi_node_hybrid_model.py