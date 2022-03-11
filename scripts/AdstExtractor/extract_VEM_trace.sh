#!/bin/bash

source /cr/data01/hahn/offline_versions/newest-stable/set_offline_env.sh

SOURCE_PATH="/lsdf/auger/tmp/hahn/test_production/napoli/ub/QGSJET-II.04/proton/19_19.5/01/*"
TARGET_DIR="/cr/users/filip/data/second_simulation/tensorflow/tmp_data/"
STEP=0

for FILE in $SOURCE_PATH
do
    # sleep condition if large amount of files in tmp_data
    while [ $(ls $TARGET_DIR | wc -l) -ge 10 ]; do sleep 2; done
    ./AdstExtractor $FILE $TARGET_DIR
    let STEP=STEP+1
    echo "STEP $STEP/9131 completed"
done
