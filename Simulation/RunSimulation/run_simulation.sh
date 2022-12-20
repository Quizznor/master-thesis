#!/bin/bash

BOOTSTRAP="/cr/users/filip/Simulation/Bootstraps/bootstrap_$4.xml"
BOOTSTRAP_SRC="/cr/users/filip/Simulation/AugerOffline/bootstrap.xml.in"
source /cr/users/filip/scripts/auger_env_22.sh


# Prepare bootstrap
INPUT='NR==59 {$0=FILE} '
PATTERN='NR==62 {$0=PATTERN1} NR==66 {$0=PATTERN2} '
OUTPUT='NR==95 {$0="/cr/tempdata01/filip/QGSJET-II/protons/"ENERGY"/root_files/"NAME".root"} '
# OUTPUT='NR==95 {$0="/cr/users/filip/Simulation/TestOutput/"NAME".root"} '
SEED='NR==119 {$0=DSEED} NR==122 {$0=PSEED} { print }'
AWK_CMD=$INPUT$PATTERN$OUTPUT$SEED

awk -v FILE=$2$1 -v PATTERN1="$7" -v PATTERN2="$8" -v NAME=$4 -v ENERGY=$5 -v DSEED=$6 -v PSEED="00000$(( $6 + 1 ))" "$AWK_CMD" $BOOTSTRAP_SRC > $BOOTSTRAP

# Run Simulation
/cr/users/filip/Simulation/AugerOffline/userAugerOffline --bootstrap $BOOTSTRAP

# Delete bootstrap
rm -rf $BOOTSTRAP

# OUTDATED #########################################################################################################
# # # convert .root files to python-friendly csv
# mkdir /cr/tempdata01/filip/tmp_data/$1
# /cr/data01/filip/SdRecEvent/AdstExtractor/AdstComponentExtractor $ROOT_FILE /cr/tempdata01/filip/tmp_data/$1
# /cr/data01/filip/SdRecEvent/AdstExtractor/convert_VEM_trace.py /cr/tempdata01/filip/tmp_data/$1/ $3 "True"
# rm -rf /cr/tempdata01/filip/tmp_data/$1
###################################################################################################################

# New ADST Component extractor! =)
/cr/users/filip/Simulation/AdstExtractor/AdstComponentExtractor "$3/root_files/$4.root"

# Delete root file
rm -rf "$3/root_files/$4.root"