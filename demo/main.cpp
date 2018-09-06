#include "Net.h"
#include <iostream>
#include <Windows.h>

#define N 2
#define M 100

int layerNeurons[5] = {N, 2, 10, 10, 6};
std::string functions[4] = { "sigmoid", "sigmoid", "sigmoid", "sigmoid"};

int main(int arg, char** argv)
{
	std::vector<int> layerNeuronNum;
	std::vector<std::string> actvationFunctions;

	for (int i = 0; i < 5; ++i)
	{
		layerNeuronNum.push_back(layerNeurons[i]);
		if (i < 4)
		{
			actvationFunctions.push_back(functions[i]);
		}
	}

	int t = GetTickCount();

	mario::Net ann(layerNeuronNum, actvationFunctions);

	ann.load();

	ann.initANN();

	ann.train(0.5, 0.995, 100, 50000);

	std::cout << GetTickCount() - t << "ms" << std::endl;

	getchar();

	return 0;
}
