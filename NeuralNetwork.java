import java.util.Arrays;

public class NeuralNetwork {

    private Layer[] hiddenLayers;
    private Layer outputLayer;

    public NeuralNetwork(int inputSize, int hiddenSize, int outputSize) {
        hiddenLayers = new Layer[2];

        hiddenLayers[0] = new Layer(hiddenSize, inputSize);

        for (int i = 1; i < hiddenLayers.length; i++) {
            hiddenLayers[i] = new Layer(hiddenSize, hiddenSize);
        }

        outputLayer = new Layer(outputSize, hiddenSize);
    }


    public double[] forward(double[] inputs) {
        double[] hiddenOutputs = hiddenLayers[0].forward(inputs);

        for (int i = 1; i < hiddenLayers.length; i++) {
            hiddenOutputs = hiddenLayers[i].forward(hiddenOutputs);
        }

        return outputLayer.forward(hiddenOutputs);
    }


    /**
     * This is a method that is used to train our neural network
     */
    public void train(
        double[][] inputs,
        double[][] targets,
        int epochs,
        double learningRate) {

        for (int e = 0; e < epochs; e++) {
            for (int i = 0; i < inputs.length; i++) {
                double[] output = forward(inputs[i]);

                backward(inputs[i], output, targets[i]);
            }
        }

    }


    public void backward(double[] inputs, double[] output, double[] targets) {
        // made using notes from w3school

        double[] outputDeltas = new double[outputLayer.layer.length];

        for (int i = 0; i < outputDeltas.length; i++) {
            double error = targets[i] - output[i];
            outputDeltas[i] = error * outputLayer.layer[i].sigmoidDerivative(
                output[i]);
        }

        double[][] hiddenDeltas = new double[hiddenLayers.length][];

        for (int layer = hiddenLayers.length - 1; layer > 0; layer--) {
            Neuron[] neurons = hiddenLayers[layer].layer;
            double[] deltas = new double[neurons.length];

            for (int n = 0; n < neurons.length; n++) {
                if (layer == hiddenLayers.length - 1) {
                    for (int o = 0; o < outputLayer.layer.length; o++) {
                        double[] weights = outputLayer.layer[o].getWeights();

                        deltas[n] = neurons[n].sigmoidDerivative(neurons[n]
                            .getOutput()) * (weights[o] * outputDeltas[o]);
                    }
                }
                else {
                    for (int o = 0; o < hiddenLayers[layer + 1].layer.length; o++) {
                        double[] weights = hiddenLayers[layer + 1].layer[o].getWeights();
                        deltas[n] += neurons[n].sigmoidDerivative(neurons[n].getOutput()) 
                                    * weights[n] * hiddenDeltas[layer + 1][o];
                    }
                }
                hiddenDeltas[layer] = deltas;
            }
        }

        for (int i = 0; i < outputDeltas.length; i++) {
            outputLayer.layer[i].updateWeights(0.01, outputDeltas[i]);
        }
        
        for(int layer = hiddenLayers.length - 1; layer > 0; layer--) {
            Neuron[] neurons = hiddenLayers[layer].layer;
            double[] deltas = hiddenDeltas[layer];
            
            for(int n = 0; n < neurons.length; n++) {
                neurons[n].updateWeights(0.01, deltas[n]);
            }
        }

    }


    public static void main(String[] args) {
        // Define dummy inputs and targets
        double[][] inputs = { 
            { 0.0, 0.0 }, 
            { 0.0, 1.0 }, 
            { 1.0, 0.0 }, 
            { 1.0, 1.0 } 
        };

        double[][] targets = { 
            { 0.0 }, // Output for {0.0, 0.0}
            { 1.0 }, // Output for {0.0, 1.0}
            { 1.0 }, // Output for {1.0, 0.0}
            { 0.0 } // Output for {1.0, 1.0}
        };

        // Create and train the neural network
        NeuralNetwork nn = new NeuralNetwork(2, 4, 1); // 2 inputs, 4 hidden
                                                       // neurons, 1 output
        nn.train(inputs, targets, 10000, 0.01);

        // Test the network
        for (double[] input : inputs) {
            double[] output = nn.forward(input);
            System.out.println("Input: " + Arrays.toString(input)
                + " -> Predicted Output: " + Arrays.toString(output));
        }
    }

}
