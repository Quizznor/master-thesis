#!/usr/bin/env bash

if [[ ! -z "$2" ]]
then
    rm -rf /cr/work/filip/*
    cp -r /cr/users/filip/Binaries/ /cr/work/filip/
else
    echo "DO NOT OVERWRITE BINARIES in /cr/work/filip/Binaries/, are you sure about this?"
fi

condor_submit $1