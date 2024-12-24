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

        for (int i = 0; i < outputLayer.layer.length; i++) {
            double error = targets[i] - output[i];
            double totalError = error * (output[i] * (1 - output[i]));

            outputLayer.layer[i].setDelta(totalError);
        }

        for (int layer = hiddenLayers.length - 1; layer >= 0; layer--) {
            Neuron[] neurons = hiddenLayers[layer].layer;

            for (int n = 0; n < neurons.length; n++) {
                double deltaSum = 0.0;

                if (layer == hiddenLayers.length - 1) {
                    for (Neuron outputNeuron : outputLayer.layer) {
                        deltaSum += outputNeuron.getWeights()[n] * outputNeuron
                            .getDelta();
                    }
                }
                else {
                    for (Neuron nextNeuron : hiddenLayers[layer + 1].layer) {
                        deltaSum += nextNeuron.getWeights()[n] * nextNeuron
                            .getDelta();
                    }
                }

                double delta = deltaSum * (neurons[n].getOutput() * (1
                    - neurons[n].getOutput()));
                neurons[n].setDelta(delta);
            }
        }

        // Update weights for the output layer
        for (Neuron neuron : outputLayer.layer) {
            neuron.updateWeights(.9, neuron.getDelta());
        }

        // Update weights for the hidden layers
        for (int layer = hiddenLayers.length - 1; layer >= 0; layer--) {
            Neuron[] neurons = hiddenLayers[layer].layer;

            for (Neuron neuron : neurons) {
                neuron.updateWeights(.9, neuron.getDelta());
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
        NeuralNetwork nn = new NeuralNetwork(2, 4, 1); 
        nn.train(inputs, targets, 50000, 1);

        // Test the network
        for (double[] input : inputs) {
            double[] output = nn.forward(input);
            System.out.println("Input: " + Arrays.toString(input)
                + " -> Predicted Output: " + Arrays.toString(output));
        }
    }

}
