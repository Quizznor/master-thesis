#!/bin/bash

source /cr/data01/hahn/offline_versions/newest-stable/set_offline_env.sh

SOURCE_PATH="/lsdf/auger/tmp/hahn/test_production/napoli/ub/QGSJET-II.04/proton/19_19.5/01/*"
TARGET_DIR="/cr/data01/filip/second_simulation/tensorflow/tmp_data/"

for FILE in $SOURCE_PATH
do
    # sleep condition if large amount of files in tmp_data
    while [ $(ls $TARGET_DIR | wc -l) -ge 10 ]; do sleep 2; done
    /cr/users/filip/scripts/AdstExtractor/AdstExtractor $FILE $TARGET_DIR
done
