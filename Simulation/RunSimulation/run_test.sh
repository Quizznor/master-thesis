FILE="/cr/tempdata01/filip/QGSJET-II/protons/$1/root_files/$2"
DEST="/cr/users/filip/Simulation/TestShowers/root_files/$2"

cp $FILE $DEST
/cr/users/filip/Simulation/AdstExtractor/AdstComponentExtractor $DEST