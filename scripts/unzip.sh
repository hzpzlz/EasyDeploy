#!/bin/bash

rootpath='/home/hzp/datasets/night_scan/L1_project/0505_4F'
python zipmv.py --root_path $rootpath

for file in $rootpath/*
do
    echo $file
    for zips in $file/*
    do
        echo $zips
        unzip -d $file $zips
    done
done

