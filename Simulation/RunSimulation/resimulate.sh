TARGET_DIR="/cr/tempdata01/filip/QGSJET-II/protons/$1"

rm -rf $TARGET_DIR/*.csv

for FILE in /cr/tempdata01/filip/QGSJET-II/protons/$1/root_files/*
do
    ../AdstExtractor/AdstComponentExtractor $FILE
done