#!/bin/bash

for item in airplane car bird cat deer dog monkey horse ship truck; do
  echo $1/$item
  echo $2/$item.txt
  find $1/$item -name *.png > $2/$item.txt
done
find $1 -name *.png > $2/all.txt