#!/bin/bash

source /cr/data01/hahn/offline_versions/newest-stable/set_offline_env.sh
python3 /cr/users/filip/scripts/AdstExtractor/extract_VEM_trace.py "$@"
