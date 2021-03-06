#!/bin/bash
# 
# Copyright (c) 2017-2018, SyLabs, Inc. All rights reserved.
# Copyright (c) 2017, SingularityWare, LLC. All rights reserved.
#
# Copyright (c) 2015-2017, Gregory M. Kurtzer. All rights reserved.
# 
# Copyright (c) 2016, The Regents of the University of California, through
# Lawrence Berkeley National Laboratory (subject to receipt of any required
# approvals from the U.S. Dept. of Energy).  All rights reserved.
# 
# This software is licensed under a customized 3-clause BSD license.  Please
# consult LICENSE file distributed with the sources of this project regarding
# your rights to use or distribute this software.
# 
# NOTICE.  This Software was developed under funding from the U.S. Department of
# Energy and the U.S. Government consequently retains certain rights. As such,
# the U.S. Government has been granted for itself and others acting on its
# behalf a paid-up, nonexclusive, irrevocable, worldwide license in the Software
# to reproduce, distribute copies to the public, prepare derivative works, and
# perform publicly and display publicly, and to permit other to do so. 
# 
# 


prefix="/usr/local"
exec_prefix="${prefix}"
libexecdir="${exec_prefix}/libexec"
sysconfdir="/etc"
localstatedir="${prefix}/var"
bindir="${exec_prefix}/bin"

SINGULARITY_USER_NS=""
SINGULARITY_OVERLAY_FS="0"
KVERS=`uname -r`
if test -f "/lib/modules/$KVERS/modules.dep"; then
    if grep -q 'overlay.ko' "/lib/modules/$KVERS/modules.dep"; then
        SINGULARITY_OVERLAY_FS="1"
    fi
fi

SINGULARITY_libexecdir="$libexecdir"
SINGULARITY_sysconfdir="$sysconfdir"
SINGULARITY_localstatedir="$localstatedir"
SINGULARITY_PATH="$bindir"

export SINGULARITY_libexecdir SINGULARITY_sysconfdir SINGULARITY_localstatedir SINGULARITY_PATH SINGULARITY_OVERLAY_FS SINGULARITY_USER_NS



if [ -z "$CLEAN_SHELL" ]; then
    /bin/echo "Building/Installing Singularity to temporary directory"
    /bin/echo "Reinvoking in a clean shell"
    sleep 1
    # Keep docker host for travis under centos 7, which runs the whole test suite in a docker container
    exec env -i CLEAN_SHELL=1 PATH="$SINGULARITY_PATH:/bin:/usr/bin:/sbin:/usr/sbin:/usr/local/bin:/usr/local/sbin" \
                DOCKER_HOST="$DOCKER_HOST" bash "$0" "$@"
fi

if [ ! -d "tests" ]; then
    /bin/echo "ERROR: Run this from the singularity source root"
    exit 1
fi

if [ ! -x "$SINGULARITY_PATH/singularity" ]; then
    /bin/echo "ERROR: Could not locate singularity program at: $SINGULARITY_PATH/singularity"
    exit 1
fi

if [ ! -d "./tests/" ]; then
    /bin/echo "ERROR: Could not locate singularity test directory"
    exit 1
fi

if ! cd tests; then
    /bin/echo "ERROR: Could not change into the Singularity test directory"
    exit 1
fi

if [ -n "$1" ]; then
    for i in $@; do
        test=`basename "$i"`
        if [ -f "$test" ]; then
            if ! /bin/sh "$test"; then
                /bin/echo "ERROR: Failed running test: $test"
                exit 1
            fi
        else
            echo "ERROR: Could not find test: '$test'"
            exit 1
        fi
    done
else
    for test in *.sh; do
        if [ -f "$test" ]; then
            if ! /bin/sh "$test"; then
                /bin/echo "ERROR: Failed running test: $test"
                exit 1
            fi
        fi
    done
fi


echo
echo "All tests passed"
