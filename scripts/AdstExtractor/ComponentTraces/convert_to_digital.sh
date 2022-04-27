SOURCE_PATH="/cr/data01/filip/02_simulation/component_signal/*"

for FILE in $SOURCE_PATH
do
    python convert_to_digital.py $FILE
done