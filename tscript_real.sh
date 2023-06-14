rm ../data/testoutput/*.raw
rm ../data/testoutput/*.jpg
cp ../data/real/test/input/$1 ../data/real/test_/input/test.raw
cp ../data/real/test/sino/$1 ../data/real/test_/sino/test.raw
cp ../data/real/test/label/$1 ../data/real/test_/label/test.raw
filename="${1%.*}"
mat="$filename.mat"
cp ../data/real/test/matlab/$mat ../data/real/test_/matlab/test.mat
python test.py
mv ../data/testoutput/output.raw ../data/result/real/Ours/$1
