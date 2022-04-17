#!/bin/bash

source /cr/data01/hahn/offline_versions/newest-stable/set_offline_env.sh

SOURCE_PATH="/lsdf/auger/tmp/hahn/test_production/napoli/ub/QGSJET-II.04/proton/19_19.5/01/*"
TEMPORARY_DIR="/cr/data01/filip/02_simulation/tmp_data/"
TARGET_DIR="/cr/data01/filip/02_simulation/signal/"


for FILE in $SOURCE_PATH
do
    # sleep condition if large amount of files in tmp_data (not really needed anymore)
    # while [ $(ls $TEMPORARY_DIR | wc -l) -ge 10 ]; do sleep 2; done

    if ! [[ $FILE =~ "trigger_all_adst" ]]; then
        /cr/users/filip/scripts/AdstExtractor/AdstExtractor $FILE $TEMPORARY_DIR
        /cr/users/filip/scripts/AdstExtractor/convert_VEM_trace.py $TEMPORARY_DIR $TARGET_DIR
    fi
done