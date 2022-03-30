#!/usr/bin/env bash

export PATH="${PATH}:/cr/users/filip/scripts"
export PYTHONPATH="${PYTHONPATH}:/cr/users/filip/scripts"

if $REQUIRE_TENSORFLOW
then
    source /cr/data01/hahn/envs/tfenv/bin/activate
fi

echo "****************************"
echo "| Automated job submission:"
echo "| using $(echo python -V)"
echo "****************************"
echo "                            "

python /cr/users/filip/scripts/$1 ${@:2}