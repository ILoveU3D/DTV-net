rm ../data/testoutput/*.raw
rm ../data/testoutput/*.jpg
cp ../data/trainData/$1 ../data/test/test.raw
cp ../data/inputTrainData/$1 ../data/testInput/test.raw
python test.py
