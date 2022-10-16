#!/bin/bash

for item in airplane automobile bird cat deer dog frog horse ship truck; do
  echo $1/$item
  echo $2/$item.txt
  find $1/$item -name *.png > $2/$item.txt
done
find $1 -name *.png > $2/all.txt