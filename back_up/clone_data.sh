#!/bin/bash

if [ "$#" -ne 2 ]; then
    echo "$0 <target_dir> <prefix_number>"
    exit 1
fi

target_dir="$1"
prefix="$2"


if [ ! -d "$target_dir" ]; then
    echo "not exist: $target_dir"
    exit 1
fi

for file in "$target_dir"/*; do
    [ -f "$file" ] || continue

    filename=$(basename "$file")
    newname="${prefix}${filename}"

    mv "$file" "$target_dir/$newname"
done

exit 0