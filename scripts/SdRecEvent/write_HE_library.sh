#!/bin/bash

source /cr/users/filip/scripts/auger_env.sh

SOURCE_PATH_WC="/lsdf/auger/tmp/hahn/test_production/napoli/ub/QGSJET-II.04/proton/19_19.5/00/*"
SOURCE_PATH="/lsdf/auger/tmp/hahn/test_production/napoli/ub/QGSJET-II.04/proton/19_19.5/00"
WORKING_PATH="/cr/tempdata01/filip/protons/19_19.5/root_files/"

# # work entire dataset
# for FILE in $SOURCE_PATH_WC
# do   
#     if ! [[ $FILE =~ "trigger_all_adst" ]]; then

#         BASENAME="$(basename -- $FILE)"
#         echo "$SOURCE_PATH/$BASENAME"

#         cp $FILE /cr/tempdata01/filip/protons/19_19.5/root_files/
#         /cr/data01/filip/SdRecEvent/AdstExtractor/AdstComponentExtractor "$WORKING_PATH/$BASENAME"
#         rm -rf $WORKING_PATH/$BASENAME
#     fi
# done

# test one single file
FILE="/lsdf/auger/tmp/hahn/test_production/napoli/ub/QGSJET-II.04/proton/19_19.5/00/DAT807679_01_adst.root"

BASENAME="$(basename -- $FILE)"
echo "$SOURCE_PATH/$BASENAME"

cp $FILE /cr/tempdata01/filip/protons/19_19.5/root_files/
/cr/users/filip/scripts/SdRecEvent/AdstExtractor/AdstComponentExtractor "$WORKING_PATH/$BASENAME"
rm -rf $WORKING_PATH/$BASENAME