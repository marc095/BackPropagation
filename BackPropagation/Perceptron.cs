using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace BackPropagation
{
    class Perceptron
    {
        int maxEpoch = 10;
        int numInputs = 2;
        int numOutputs = 1;
        int numHiddens = 2;
        double n = 0.85; //rate of training
        double[] input;
        double[] output;
        double[] hidden;
        double[] weights;
        double[] activationOutput;
        int numberWeights;
        private double[][] ihWeights; 
        private double[][] hoWeights;
        double[] activationVals;
        public Perceptron()
        {
            this.input = new double[numInputs];
            this.output = new double[numOutputs];
            this.hidden = new double[numHiddens];
            this.numberWeights = (numInputs * numHiddens) +
          (numHiddens * numOutputs) + numHiddens + numOutputs;
            this.weights = new double[numberWeights];
            this.ihWeights = this.Matrix(numInputs + 1, numHiddens);
            this.hoWeights = this.Matrix(numHiddens + 1 , numOutputs);
            this.activationOutput = new double[numOutputs];
            activationVals = new double[numInputs + 1];
            InitializeWieghts();
            SetWeights();
        }

        private double[][] Matrix(int input, int count) 
        {
            double[][] result = new double[input][];
            for (int i = 0; i < input; i++)
            {
                result[i] = new double[count];
            }
            return result;
        }

        private void InitializeWieghts()
        {
            Random rand = new Random(1);
            for (int i = 0; i < weights.Length; i++)
            {
                weights[i] = 0.01* rand.NextDouble();
            }
        }

        private void Compute() 
        {
            double[] net = new double[numInputs + 1];

            for (int i = 0; i < numHiddens + 1; i++)
            {
                for (int j = 0; j < numInputs; j++)
                {
                    net[i] += this.input[j] * this.ihWeights[i][j];
                }
            }

            for (int i = 0; i < net.Length; i++)
            {
                activationVals[i] = SoftMax(net[i]);
            }

            for (int j = 0; j < numHiddens + 1; j++)
            {
                for (int i = 0; i < output.Length; i++)
                {
                    output[i] += hoWeights[j][i] * activationVals[j];
                }
            }
            for (int i = 0; i < output.Length; i++)
            {
                activationOutput[i] = SoftMax(output[i]);
            }
        }

        public void ThresholdCompute()
        {
            double[] net = new double[numInputs+1];
            double[] thresholdVals = new double[net.Length];
            
            for (int i = 0; i < numHiddens + 1; i++)
            {
                for (int j = 0; j < numInputs; j++)
                {
                    net[i] += this.input[j]*this.ihWeights[i][j];
                }
            }

            for (int i = 0; i < net.Length; i++)
            {
                thresholdVals[i] = Threshold(net[i]);
            }

            for (int j = 0; j < numHiddens + 1; j++)
			{
            for (int i = 0; i < output.Length; i++)
            {
                output[i] += hoWeights[j][i]*thresholdVals[j]; 
            }   
          }
        }

        private double Threshold(double val) 
        {
            if (val > 0.5) return 1.0;
            return 0.0;
        }

        private double SoftMax(double x)
        {
        return 1.0 /(1 + Math.Exp(-x));
        }

        public void Train() 
        {
            double[] delta = new double[numInputs + 1];
            double[][] trainSample = Matrix(numInputs + 1, 3);
            double [] outputSignals = new double[numOutputs];
            double [] hiddenSignals = new double[numHiddens];

            trainSample[0][0] = 0;
            trainSample[0][1] = 0;
            trainSample[0][2] = 0;

            trainSample[1][0] = 1;
            trainSample[1][1] = 0;
            trainSample[1][2] = 1;

            trainSample[2][0] = 0;
            trainSample[2][1] = 1;
            trainSample[2][2] = 1;

            double[] results = new double[numOutputs];
            int epoch = 0;
            double error = 0;
            double derivative = 0;
            int errorInterval = maxEpoch / 10;
            while(epoch < maxEpoch)
            {
                ++epoch;

                if (epoch % errorInterval == 0)
                {
                    double trainError = MeanSquaredError(trainSample);
                    Console.WriteLine("epoch = {0} | error = {1}", epoch,trainError);                   
                    break;
                }

                for (int i = 0; i < trainSample.Length; i++)
                {
                    Array.Copy(trainSample[i], input, numInputs);
                    Array.Copy(trainSample[i], numOutputs+1, results, 0, numOutputs);
                    Compute();
        
                    for (int j = 0; j < numOutputs; j++)
                    {
                        error = results[j] - activationOutput[j];
                        derivative = (1 - activationOutput[j]) * activationOutput[j];
                        outputSignals[j] = error * derivative;
                    }
         
                    for (int j = 0; j < numHiddens + 1; j++)
                    {
                        for(int k = 0; k < numOutputs; k++){
                            derivative = activationVals[j]*(1 - activationVals[j]);
                            delta[j] += outputSignals[k]*hoWeights[j][k] * derivative;
                       }
                    }
                    double newWeights = 0;
                    //update input-hidden weights
                    for (int j = 0; j < numInputs + 1; j++)
                    {
                        for (int k = 0; k < numHiddens; k++)
                        {
                                newWeights = input[k] * (delta[j] * n);
                                ihWeights[j][k] = newWeights;
                        }                        
                    }
                    //update hidden-output weights
                    for (int j = 0; j < numHiddens + 1; j++)
                    {
                        for (int k = 0; k < numOutputs; k++)
                        {
                            newWeights = activationVals[j]*(outputSignals[k]*n);
                            hoWeights[j][k] = newWeights;
                        }
                    }
                }
            }
        }
        private double MeanSquaredError(double[][] train)
        {
            double SquaredError = 0.0;
            double[] result = new double[numOutputs];

            for (int i = 0; i < train.Length; ++i)
            {
                Array.Copy(train[i], input, numInputs);
                Array.Copy(train[i], numInputs, result, 0, numOutputs);
                Compute();
                for (int j = 0; j < numOutputs; ++j)
                {
                    double err = result[j] - this.activationOutput[j];
                    SquaredError += err * err;
                }
            }
            return SquaredError / train.Length;
        } 

        private void SetWeights()
        {
            int count = 0;
            for (int i = 0; i < numInputs + 1; i++)
            {
                for (int j = 0; j < numHiddens; j++)
                {
                    this.ihWeights[i][j] = weights[count++];
                }
            }
            for (int i = 0; i < numHiddens +1; i++)
            {
                for (int j = 0; j < numOutputs; j++)
                {
                    this.hoWeights[i][j] = this.weights[count++];
                }
            }
        }
    }
}
