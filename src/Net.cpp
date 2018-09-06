#include "Net.h"
#include <stdlib.h>
#include <math.h>

#define N 2
#define M 100

namespace mario
{
	/*
	double getRandom(double start, double end)
	{
		int n = rand() % 1000 - 500;	//-500~499
		
		return n / double(500.0);
	}
	*/

	Eigen::MatrixXd sigmoid(const Eigen::MatrixXd &input)
	{
		Eigen::MatrixXd out(input.rows(), input.cols());

		for (int i = 0; i < input.rows(); ++i)
		{
			for (int j = 0; j < input.cols(); ++j)
			{
				double n = input(i, j);
				out(i, j) = 1 / (1 + exp(-1 * n));
			}

		}

		return out;
	}

	Eigen::MatrixXd relu(const Eigen::MatrixXd &input)
	{
		Eigen::MatrixXd out(input.rows(), input.cols());
		Eigen::MatrixXd zeros = Eigen::MatrixXd::Zero(input.rows(), input.cols());

		out = zeros.cwiseMax(input);
		/*
		for (int i = 0; i < input.rows(); ++i)
		{
			for(int j = 0; j < input.cols(); ++j)
			{
				out(i, j) = input(i, j) < 0. ? 0. : input(i, j);
			}
		}
		*/
		return out;
	}

	Eigen::MatrixXd tanh(const Eigen::MatrixXd &input)
	{
		Eigen::MatrixXd out(input.rows(), input.cols());

		double n, m;
		for (int i = 0; i < input.rows(); ++i)
		{
			for (int j = 0; j < input.cols(); ++j)
			{
				double e = input(i, j);
				n = exp(e);
				m = exp(-1 * e);
				out(i, j) = (n - m) / (n + m);
			}
			
		}

		return out;
	}

	void Net::initWeights()
	{
		m_weights.clear();

		for (unsigned int i = 0; i<m_layerNeuronNum.size() - 1; ++i)
		{
			m_weights.push_back(Eigen::MatrixXd::Random(m_layerNeuronNum[i + 1], m_layerNeuronNum[i]));
			//m_weights.push_back(Eigen::MatrixXd::Ones(m_layerNeuronNum[i + 1], m_layerNeuronNum[i]));
		}
	}

	void Net::initBias()
	{
		m_bias.clear();

		for (unsigned int i = 0; i<m_layerNeuronNum.size() - 1; ++i)
		{
			m_bias.push_back(Eigen::MatrixXd::Random(m_layerNeuronNum[i + 1], 1));
			//m_bias.push_back(Eigen::MatrixXd::Ones(m_layerNeuronNum[i + 1], 1));
		}
	}

	//init之前要load输入和目标输出！
	void Net::initNeurons()
	{
		m_neurons.clear();

		m_neurons.push_back(m_input);

		for (unsigned int i = 1; i<m_layerNeuronNum.size(); ++i)
		{
			m_neurons.push_back(Eigen::MatrixXd::Random(m_layerNeuronNum[i], m_input.cols()));
			//m_neurons.push_back(Eigen::MatrixXd::Ones(m_layerNeuronNum[i], 1));
		}
	}

	void Net::initDeltaWeights()
	{
		m_deltaWeights.clear();

		for (unsigned int i = 0; i < m_layerNeuronNum.size() - 1; ++i)
		{
			m_deltaWeights.push_back(Eigen::MatrixXd::Random(m_layerNeuronNum[i + 1], m_layerNeuronNum[i]));
			//m_deltaWeights.push_back(Eigen::MatrixXd::Ones(m_layerNeuronNum[i + 1], m_layerNeuronNum[i]));
		}
	}

	void Net::initDeltaError()
	{
		m_deltaError.clear();

		for (unsigned int i = 0; i < m_layerNeuronNum.size() - 1; ++i)
		{
			m_deltaError.push_back(Eigen::MatrixXd::Random(m_layerNeuronNum[i + 1], 1));
		}
	}

	void Net::initDerivativeNeurons()
	{
		m_derivativeNeurons.clear();

		for (unsigned int i = 0; i < m_layerNeuronNum.size() - 1; ++i)
		{	
			m_derivativeNeurons.push_back(Eigen::MatrixXd::Random(m_layerNeuronNum[i + 1], 1));
			/*
			Eigen::MatrixXd e = m_neurons[i + 1];

			if ("sigmoid" == m_actvationFunctions[i])
			{
				m_derivativeNeurons.push_back(e.cwiseProduct(Eigen::MatrixXd::Ones(e.rows(), e.cols()) - e));
			}
			else if ("tanh" == m_actvationFunctions[i])
			{
				m_derivativeNeurons.push_back(Eigen::MatrixXd::Ones(e.rows(), e.cols()) - e.cwiseProduct(e));
			}
			else if ("relu" == m_actvationFunctions[i])
			{
				Eigen::MatrixXd m(m_neurons[i + 1].rows(), m_neurons[i + 1].cols());

				for (int i = 0; i < e.rows(); ++i)
				{
					if (0 >= e(i, 0))
					{
						m(i, 0) = 0;
					}
					else
					{
						m(i, 0) = 1.;
					}
				}

				m_derivativeNeurons.push_back(m);
			}*/
		}
	}

	void Net::initDeltaBias()
	{
		m_deltaBias.clear();

		for (unsigned int i = 0; i < m_layerNeuronNum.size() - 1; ++i)
		{
			m_deltaBias.push_back(Eigen::MatrixXd::Random(m_layerNeuronNum[i + 1], 1));
		}
	}

	void Net::initANN()
	{
		initNeurons();
		initWeights();
		initBias();
		initDeltaWeights();
		initDerivativeNeurons();
		initDeltaError();
		initDeltaBias();
	}
	
	Eigen::MatrixXd Net::activationFunction(const Eigen::MatrixXd &input, const std::string &functionName)
	{
		Eigen::MatrixXd out(input.rows(), input.cols());

		if ("sigmoid" == functionName)
		{
			out = sigmoid(input);
		}
		else if ("tanh" == functionName)
		{
			out = tanh(input);
		}
		else if ("relu" == functionName)
		{
			out = relu(input);
		}
		else
		{
			std::cout << "function name is wrong" << std::endl;
		}

		return out;
	}

	void Net::forward()
	{
		for (unsigned int i = 0; i < m_layerNeuronNum.size() - 1; ++i)
		{
			Eigen::MatrixXd bias = Eigen::MatrixXd::Random(m_bias[i].rows(), m_input.cols());

			for (int j = 0; j < m_input.cols(); ++j)
			{
				bias.col(j) = m_bias[i];
			}

			Eigen::MatrixXd product = m_weights[i] * m_neurons[i] +bias;
			m_neurons[i + 1] = activationFunction(product, m_actvationFunctions[i]);

			//计算神经元的导数m_derivativeNeurons，从第1层到输出层（索引是从0开始的）
			Eigen::MatrixXd fx = m_neurons[i + 1];

			if ("sigmoid" == m_actvationFunctions[i])
			{
				//fx * (1 - fx)
				m_derivativeNeurons[i] = fx.cwiseProduct(Eigen::MatrixXd::Ones(fx.rows(), fx.cols()) - fx).rowwise().mean();
			}
			else if ("tanh" == m_actvationFunctions[i])
			{
				//1-fx^2
				m_derivativeNeurons[i] = (Eigen::MatrixXd::Ones(fx.rows(), fx.cols()) - fx.cwiseProduct(fx)).rowwise().mean();
			}
			else if ("relu" == m_actvationFunctions[i])
			{
				Eigen::MatrixXd m(m_neurons[i + 1].rows(), m_neurons[i + 1].cols());

				//	if	fx > 0 then: m_derivativeNeuron = 1
				//	if	fx < 0 then: m_derivativeNeuron = 0
				for (int j = 0; j < fx.rows(); ++j)
				{
					for (int k = 0; k < fx.cols(); ++k)
					{
						if (0. > fx(j, k))
						{
							m(j, k) = 0.;
						}
						else
						{
							m(j, k) = 1.;
						}
					}
				}

				m_derivativeNeurons[i] = m.rowwise().mean();
			}
		}

		//计算softmax输出
		int lastLayerIndex = m_layerNeuronNum.size() - 1;
		Eigen::MatrixXd lastLayer = m_neurons[lastLayerIndex];

		for (int i = 0; i < lastLayer.rows(); ++i)
		{
			for (int j = 0; j < lastLayer.cols(); ++j)
			{
				lastLayer(i, j) = exp(lastLayer(i, j));
			}
		}

//		double sum = lastLayer.sum();
		m_neurons[lastLayerIndex] = lastLayer;
	}

	void Net::calcDeltaError()
	{
		//输出层的deltaError
		m_deltaError[m_neurons.size() - 2] = m_neurons.back().rowwise().mean() - m_targetOutput;
		m_loss = m_deltaError[m_neurons.size() - 2].mean();

		//有size-1个误差点，去掉最后一个，还剩size-2个
		for (int i = m_neurons.size() - 3; i >= 0; --i)
		{
//			std::cout << m_deltaError[i].rows() << " x " << m_deltaError[i].cols() << std::endl;
//			std::cout << m_weights[i + 1].rows() << " x " << m_weights[i + 1].cols() << std::endl;
//			std::cout << m_deltaError[i + 1].rows() << " x " << m_deltaError[i + 1].cols() << std::endl << std::endl;
			m_deltaError[i] = m_weights[i + 1].transpose() * m_deltaError[i + 1];
		}
	}

	void Net::calcGradient()
	{
		for (unsigned int i = 0; i < m_layerNeuronNum.size() - 1; ++i)
		{
			m_deltaBias[i] = m_deltaError[i];

//			std::cout << m_deltaWeights[i].rows() << " x " << m_deltaWeights[i].cols() << std::endl;
//			std::cout << m_deltaError[i].rows() << " x " << m_deltaError[i].cols() << std::endl;
//			std::cout << m_neurons[i].rows() << " x " << m_neurons[i].cols() << std::endl << std::endl;

			m_deltaWeights[i] = m_deltaError[i] * m_neurons[i].rowwise().mean().transpose();
		}
	}

	void Net::updateParams()
	{
		for (unsigned int i = 0; i < m_layerNeuronNum.size() - 1; ++i)
		{
			m_weights[i] -= m_learningRate * m_deltaWeights[i];
			m_bias[i] -= m_learningRate * m_deltaBias[i];
		}
	}

	void Net::load()
	{
		m_input = Eigen::MatrixXd::Random(N, M) * 10;
		m_targetOutput = Eigen::MatrixXd::Ones(m_layerNeuronNum.back(), 1);
	}

	void Net::train(double learninigRate, double decay, int decayNum, int epoch)
	{	
		m_learningRate = learninigRate;

		for (int i = 1; i < epoch + 1; ++i)
		{
			forward();
			calcDeltaError();
			calcGradient();
			updateParams();
			
			if (0 == i % 500)
			{
				std::cout << i << ":\t\t" << m_loss << std::endl;
			}
			else if (1 == i)
			{
				std::cout << i << ":\t\t" << m_loss << std::endl;
			}
//			std::cout << m_deltaWeights[0] << std::endl;

			if (0 == (i % decayNum))
			{
				m_learningRate *= decay;
			}
		}
	}

}
