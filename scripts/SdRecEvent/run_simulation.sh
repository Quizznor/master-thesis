#!/bin/bash

BOOTSTRAP="/cr/data01/filip/SdRecEvent/bootstrap_$1.xml"
# ROOT_FILE="/cr/tempdata01/filip/protons/16.5_17/root_files/$1.root"
ROOT_FILE="/lsdf/auger/tmp/hahn/test_production/napoli/ub/QGSJET-II.04/proton/19_19.5/00/$1__adst.root"
TARGET="/cr/tempdata01/filip/protons/19_19.5/"
SOURCE_PATH=$2$1
source /cr/users/filip/scripts/auger_env.sh

# # Prepare bootstrap don't forget to change directories in awk!!!!
# awk -v FILE=$SOURCE_PATH -v NAME=$1 'NR==54 {$0=FILE} NR==86 {$0="/cr/tempdata01/filip/protons/16.5_17/root_files/"NAME".root"} { print }' /cr/data01/filip/SdRecEvent/bootstrap.xml.in > $BOOTSTRAP

# # Run Simulation
# /cr/data01/filip/SdRecEvent/userAugerOffline --bootstrap $BOOTSTRAP

# # Delete bootstrap
# rm -rf $BOOTSTRAP

# # # convert .root files to python-friendly csv
# mkdir /cr/tempdata01/filip/tmp_data/$1
# /cr/data01/filip/SdRecEvent/AdstExtractor/AdstComponentExtractor $ROOT_FILE /cr/tempdata01/filip/tmp_data/$1
# /cr/data01/filip/SdRecEvent/AdstExtractor/convert_VEM_trace.py /cr/tempdata01/filip/tmp_data/$1/ $TARGET "True"
# rm -rf /cr/tempdata01/filip/tmp_data/$1
# rm -rf $ROOT_FILE

# New ADST Component extractor! =)
echo $ROOT_FILE
echo $TARGET
/cr/data01/filip/SdRecEvent/AdstExtractor/AdstComponentExtractor $ROOT_FILE $TARGET