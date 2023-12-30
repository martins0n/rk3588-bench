compile:
	g++ bench.cpp -o program -O3 -std=c++14 -I./librknn_api/include -L./librknn_api/aarch64 -lrknnrt -lopenblas