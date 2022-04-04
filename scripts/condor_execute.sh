#!/bin/bash

case $(head $1 -n 1) in
*"sh"*)

    export PATH="${PATH}:/cr/users/filip/scripts"
    BASH_VERSION="$(bash --version | awk 'FNR==1 {print $4}')"

    echo "*******************************"
    echo "| Automated job submission:"
    echo "| using bash $BASH_VERSION"
    echo "*******************************"
    echo ""

    INTERPRETER=bash
    ;;

*"py"*)
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
    echo ""

    INTERPRETER=python
    ;;
*)
    echo "*************************************************"
    echo "| PLEASE PROVIDE A VALID INTERPRETER (py, sh)"
    echo "| BY ADDING A SHEBANG TO YOUR EXECUTABLE FILE"
    echo "*************************************************"
    echo ""
    echo "PROCESS ABORTED WITH EXIT STATUS 1"
    exit 1
esac

echo "** START OF PROCESS OUTPUT ******************************************************"
echo ""

$INTERPRETER $1 ${@:2}

echo ""
echo "** END OF PROCESS OUTPUT ********************************************************"
echo ""
echo "Process exited with status $?"