#!/bin/bash


BOOTSTRAP="/cr/users/filip/scripts/BuildSimulation/bootstrap_$1.xml"
TARGET="$3"
ADSTNAME="$4"
SOURCE_PATH=$2$1
source /cr/users/filip/scripts/auger_env.sh

# Prepare bootstrap: don't forget to change directories in awk!!!!
awk -v FILE=$SOURCE_PATH -v NAME=$4 'NR==54 {$0=FILE} NR==86 {$0="/cr/tempdata01/filip/QGSJET-II/protons/19_19.5/root_files/"NAME".root"} { print }' /cr/users/filip/scripts/BuildSimulation/bootstrap.xml.in > $BOOTSTRAP

# Run Simulation
/cr/users/filip/scripts/BuildSimulation/userAugerOffline --bootstrap $BOOTSTRAP

# Delete bootstrap
rm -rf $BOOTSTRAP

# OUTDATED #########################################################################################################
# # # # convert .root files to python-friendly csv
# # mkdir /cr/tempdata01/filip/tmp_data/$1
# # /cr/data01/filip/SdRecEvent/AdstExtractor/AdstComponentExtractor $ROOT_FILE /cr/tempdata01/filip/tmp_data/$1
# # /cr/data01/filip/SdRecEvent/AdstExtractor/convert_VEM_trace.py /cr/tempdata01/filip/tmp_data/$1/ $TARGET "True"
# # rm -rf /cr/tempdata01/filip/tmp_data/$1
####################################################################################################################

# # New ADST Component extractor! =)
/cr/users/filip/scripts/BuildSimulation/AdstExtractor/AdstComponentExtractor "$TARGET/root_files/$ADSTNAME.root"