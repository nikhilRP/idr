echo "Compiling application..."
cd ..
mkdir bin
cd bin
cmake ../
make
./idr -a ../test/data/peak1 -b ../test/data/peak2 -g ../test/data/genome_table.txt
echo "Done compiling application :)"
