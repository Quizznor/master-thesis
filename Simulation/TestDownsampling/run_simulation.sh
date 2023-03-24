#!/bin/bash

source /cr/users/filip/Simulation/.env/auger_env.sh

# UUB
BOOTSTRAP="/cr/users/filip/Simulation/Bootstraps/bootstrap_$4.xml"
BOOTSTRAP_SRC="/cr/users/filip/Simulation/TestDownsampling/bootstrap_UUB.xml.in"

# Prepare bootstrap
INPUT='NR==59 {$0=FILE} '
PATTERN='NR==62 {$0=PATTERN1} NR==66 {$0=PATTERN2} '
OUTPUT='NR==81 {$0="/cr/tempdata01/filip/QGSJET-II/DOWNSAMPLING/UUB/root_files/"NAME".root"} '
SEED='NR==98 {$0=DSEED} NR==101 {$0=PSEED} { print }'
AWK_CMD=$INPUT$PATTERN$OUTPUT$SEED

# Prepare bootstrap
awk -v FILE=$2$1 -v PATTERN1="$7" -v PATTERN2="$8" -v NAME=$4 -v ENERGY=$5 -v DSEED=$6 -v PSEED="00000$6" "$AWK_CMD" $BOOTSTRAP_SRC > $BOOTSTRAP

# Run Simulation
/cr/users/filip/Simulation/AugerOffline/userAugerOffline --bootstrap $BOOTSTRAP

# # Delete bootstrap
# rm -rf $BOOTSTRAP

# New ADST Component extractor! =)
/cr/users/filip/Simulation/AdstReader/AdstReader 0 $3/UUB/root_files/$4.root


# UB
BOOTSTRAP="/cr/users/filip/Simulation/Bootstraps/bootstrap_$4.xml"
BOOTSTRAP_SRC="/cr/users/filip/Simulation/TestDownsampling/bootstrap_UB.xml.in"

# Prepare bootstrap
INPUT='NR==59 {$0=FILE} '
PATTERN='NR==62 {$0=PATTERN1} NR==66 {$0=PATTERN2} '
OUTPUT='NR==81 {$0="/cr/tempdata01/filip/QGSJET-II/DOWNSAMPLING/UB/root_files/"NAME".root"} '
SEED='NR==98 {$0=DSEED} NR==101 {$0=PSEED} { print }'
AWK_CMD=$INPUT$PATTERN$OUTPUT$SEED

# Prepare bootstrap
awk -v FILE=$2$1 -v PATTERN1="$7" -v PATTERN2="$8" -v NAME=$4 -v ENERGY=$5 -v DSEED=$6 -v PSEED="00000$6" "$AWK_CMD" $BOOTSTRAP_SRC > $BOOTSTRAP

# Run Simulation
/cr/users/filip/Simulation/AugerOffline/userAugerOffline --bootstrap $BOOTSTRAP

# # Delete bootstrap
# rm -rf $BOOTSTRAP

# # New ADST Component extractor! =)
/cr/users/filip/Simulation/TestDownsampling/extractUB $3/UB/root_files/$4.root