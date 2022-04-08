#!/bin/bash

function set_condor_defaults {
    export REQUIRE_TENSORFLOW=false
    export CONDOR_MEMORY="1G"
    export CONDOR_QUEUE="1"
}

function show_help {
    echo ""
    echo "USAGE: condor_submit <executable> [KWARGS]               "
    echo " Available keywords are:                                 "
    echo "  --request_memory | -m   1G  -- request 1G of RAM       "
    echo "  --repeat_n_times | -r   10  -- queue run 10 times      "
    echo "  --tensorflow     | -tf      -- require crc2.ikp.kit.edu"
    echo "  --arguments      | -a   ..  -- arguments for executable"
    echo "                                                         "
    echo "  Please provide --arguments last !!                     "
    echo "                                                         "
}

# check for valid executable
if [ ! -f $(pwd)/$1 ]
then
    show_help
    exit 1
else
    # TODO: check for file permission (i.e. is it executable by condor?)
    export CONDOR_EXECUTABLE="$1"
    shift
fi

set_condor_defaults

# fetch keyword arguments
while [ $# -gt 0 ]; do
  case "$1" in
    -m | --request_memory )
        export CONDOR_MEMORY="$2"
        shift 2
        ;;
    -r | --repeat_n_times )
        export CONDOR_QUEUE="$2"
        shift 2
        ;;
    -tf | --tensorflow )
        export REQUIRE_TENSORFLOW=true
        shift
        ;;
    -a | --arguments )
        export CONDOR_ARGUMENTS="$@"
        shift $#
        ;;
    *)
      show_help
      exit 1

  esac
done

# create job binaries for condor
SCRIPTNAME=$(echo ${CONDOR_EXECUTABLE::-3} | rev | cut -f1 -d/ | rev)
JOB_FOLDER="$HOME/condor_binaries/$SCRIPTNAME"
mkdir -p $JOB_FOLDER
RUN_NO=$(( $(ls $JOB_FOLDER | wc -l) + 1))
mkdir -p $JOB_FOLDER/run_$RUN_NO
touch $JOB_FOLDER/run_$RUN_NO/file.sub

echo "JobBatchName    = trigger_studies-$SCRIPTNAME               " >> $JOB_FOLDER/run_$RUN_NO/file.sub
echo "Executable      = $HOME/scripts/condor_execute.sh           " >> $JOB_FOLDER/run_$RUN_NO/file.sub
echo "                                                            " >> $JOB_FOLDER/run_$RUN_NO/file.sub
echo "error           = $JOB_FOLDER/run_$RUN_NO/errors.txt        " >> $JOB_FOLDER/run_$RUN_NO/file.sub
echo "output          = $JOB_FOLDER/run_$RUN_NO/output.txt        " >> $JOB_FOLDER/run_$RUN_NO/file.sub
echo "log             = $JOB_FOLDER/run_$RUN_NO/syslog.txt        " >> $JOB_FOLDER/run_$RUN_NO/file.sub
echo "                                                            " >> $JOB_FOLDER/run_$RUN_NO/file.sub
echo "arguments       = $CONDOR_EXECUTABLE $CONDOR_ARGUMENTS      " >> $JOB_FOLDER/run_$RUN_NO/file.sub

if $REQUIRE_TENSORFLOW; then
    echo "Requirements    = (TARGET.Machine == 'crc2.ikp.kit.edu')" >> $JOB_FOLDER/run_$RUN_NO/file.sub
fi

echo "request_memory  = $CONDOR_MEMORY                            " >> $JOB_FOLDER/run_$RUN_NO/file.sub
echo "                                                            " >> $JOB_FOLDER/run_$RUN_NO/file.sub
echo "should_transfer_files = YES                                 " >> $JOB_FOLDER/run_$RUN_NO/file.sub

echo "transfer_input_files  = $(pwd)/$CONDOR_EXECUTABLE, /cr/users/filip/scripts/condor_execute.sh" >> $JOB_FOLDER/run_$RUN_NO/file.sub

echo "                                                            " >> $JOB_FOLDER/run_$RUN_NO/file.sub
echo "queue $CONDOR_QUEUE                                         " >> $JOB_FOLDER/run_$RUN_NO/file.sub

condor_submit $JOB_FOLDER/run_$RUN_NO/file.sub