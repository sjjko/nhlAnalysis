#!/bin/bash

#this files converts data files to .csv files

DATAFILES=$(find -name "*.dat")


for FILES in $DATAFILES 
do
  echo "replace delimiter in file: " $FILES
   FILENAME=$(basename $FILES) 
  FILENAME_WO_ENDING=${FILENAME%%.*}
    OUTNAMEHEADER=$FILENAME_WO_ENDING"_wH.data" 
   echo "now insert header line and create new file " $OUTNAMEHEADER
    awk 'BEGIN{print "ID\t","SEASON\t","DATE\t","TEAM1\t","TEAM2\t","WON\t","SCORE\t","SHOTS\t","FACEOFF\t","TAKEAWAY\t","GIVEAWAY\t","PIM\t","HITS\t","PPG\t","ATTENDANCE\t"}{print $0}' $FILES > $OUTNAMEHEADER
  OUTNAME=$FILENAME_WO_ENDING".csv" 
  echo "new file " $OUTNAME
    sed 's/\t/,/g'  $OUTNAMEHEADER > $OUTNAME
done



