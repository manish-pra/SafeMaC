#!/bin/bash
addr="../experiments/GP_copy"

ls -d -- */
for folder in ls -d -- "$addr"/*/*/*/
do
    echo $folder
    # rm -R $folder
done