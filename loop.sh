#! /bin/sh

# script for re-running k-means until it returns 0

$@

while [ "$?" -ne "0" ];
    do $@
done
