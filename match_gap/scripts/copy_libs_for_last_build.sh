#!/bin/sh
cp $2/gap9_lib/Makefile $1
cp $2/gap9_lib/gdb_script.sh $1
#cp -r BUILD $1/
cp $2/dory_lib $1/dory -R
cp $2/sparse_lib/hw_version $1/sparse -R
cp $2/gap9_lib/include/* $1/include/
cp $2/gap9_lib/src/* $1/src/