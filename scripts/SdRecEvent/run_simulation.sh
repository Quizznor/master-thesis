#!/bin/bash

BOOTSTRAP="/cr/users/filip/scripts/SdRecEvent/bootstrap_$1.xml"
TARGET="/cr/tempdata01/filip/protons/16.5_17/"
SOURCE_PATH=$2$1
source /cr/users/filip/scripts/auger_env.sh

# Prepare bootstrap don't forget to change directories in awk!!!!
awk -v FILE=$SOURCE_PATH -v NAME=$1 'NR==54 {$0=FILE} NR==86 {$0="/cr/tempdata01/filip/protons/16.5_17/root_files/"NAME".root"} { print }' /cr/users/filip/scripts/SdRecEvent/bootstrap.xml.in > $BOOTSTRAP

# Run Simulation
/cr/users/filip/scripts/SdRecEvent/userAugerOffline --bootstrap $BOOTSTRAP

# Delete bootstrap
# rm -rf $BOOTSTRAP

# # # # convert .root files to python-friendly csv
# # mkdir /cr/tempdata01/filip/tmp_data/$1
# # /cr/data01/filip/SdRecEvent/AdstExtractor/AdstComponentExtractor $ROOT_FILE /cr/tempdata01/filip/tmp_data/$1
# # /cr/data01/filip/SdRecEvent/AdstExtractor/convert_VEM_trace.py /cr/tempdata01/filip/tmp_data/$1/ $TARGET "True"
# # rm -rf /cr/tempdata01/filip/tmp_data/$1

# # New ADST Component extractor! =)
# echo $ROOT_FILE
# echo $TARGET
# /cr/data01/filip/SdRecEvent/AdstExtractor/AdstComponentExtractor $ROOT_FILE $TARGET