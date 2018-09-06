/*
		Author:		He Zhichao
		date:			2018.09.05
		version:		v0.1
		E-mail:			mario.he@goertek.com
*/

/*
		本代码使用Eigen矩阵运算库实现多层全连接神经网络的
	前向传播（forward）与反向传播（backpropagation）。
*/

#ifndef NET_H
#define NET_H

#pragma once

#include <iostream>
#include <vector>
#include "Dense"

namespace mario
{
	class Net
	{
	public:
	//private
		std::vector<int> m_layerNeuronNum;
		std::vector<std::string> m_actvationFunctions;

		std::vector<Eigen::MatrixXd> m_neurons;
		std::vector<Eigen::MatrixXd> m_weights;
		std::vector<Eigen::MatrixXd> m_bias;

		//第i层神经元的导数与deltaError保存在第i-1个vector
		std::vector<Eigen::MatrixXd> m_deltaWeights;
		std::vector<Eigen::MatrixXd> m_deltaBias;
		std::vector<Eigen::MatrixXd> m_derivativeNeurons;
		std::vector<Eigen::MatrixXd> m_deltaError;

		Eigen::MatrixXd m_targetOutput;
		Eigen::MatrixXd m_input;

		double m_learningRate;
		double m_loss;

	protected:
		void initWeights();			//初始化权重
		void initBias();					//初始化偏置
		void initNeurons();			//初始化神经元
		void initDeltaWeights();					//初始化梯度
		void initDeltaBias();
		void initDerivativeNeurons();			//初始化神经元（激活函数）的导数，从第二（索引为1）层开始<前向传播的时候计算激活函数导数
		void initDeltaError();
		Eigen::MatrixXd activationFunction(const Eigen::MatrixXd &input, const std::string &actvationFuntion);

	public:
		Net(const std::vector<int> &layerNeuronNum, const std::vector<std::string> &activationFunctions)
			: m_layerNeuronNum(layerNeuronNum), m_actvationFunctions(activationFunctions)
		{
			if (layerNeuronNum.size() != (activationFunctions.size() + 1))
			{
				std::cout << "layerNeuronNum.size != activationFunctions.size" << std::endl;
			}

			if (layerNeuronNum.size() <= 1)
			{
				std::cout << "layerNeuronNum.size() <= 1" << std::endl;
			}

			//Debug
		}

		void initANN();
		
		//在计算前向传播中，顺便把每个神经元激活函数的导数即变量 m_derivativeNeurons 求出来了
		void forward();	

		void calcDeltaError();

		void calcGradient();

		void updateParams();

		void load();

		void train(double learninigRate = 0.3, double decay = 0.5, int decayNum = 100, int epoch = 1000);
	};	//class Net
}

#endif	//NET_H
