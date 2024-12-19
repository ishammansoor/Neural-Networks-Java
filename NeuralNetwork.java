
public class NeuralNetwork {
    
    private Layer[] hiddenLayers;
    private Layer outputLayer;
    
    public NeuralNetwork(int inputSize, int hiddenSize, int outputSize) {
        hiddenLayers = new Layer[2]; 
        
        hiddenLayers[0] = new Layer(hiddenSize, inputSize);
        
        for(int i = 1; i < hiddenLayers.length; i++) {
            hiddenLayers[i] = new Layer(hiddenSize, hiddenSize);
        }
        
        outputLayer = new Layer(outputSize, hiddenSize); // 1 neuron in the output layer, 16 inputs from hidden layer
    }
    
    public double[] forward(double[] inputs) {
        double[] hiddenOutputs = hiddenLayers[0].forward(inputs);
        
        for(int i = 1; i < hiddenLayers.length; i++) {
            hiddenOutputs = hiddenLayers[i].forward(hiddenOutputs);
        }
          
        return outputLayer.forward(hiddenOutputs);
    }


    public static void main(String[] args) {
        
        
        
        double[] inputs = {0.5, 0.8};
        NeuralNetwork nn = new NeuralNetwork(inputs.length, 16, 10);
        double[] output = nn.forward(inputs);
        
        System.out.println("Predicted Output: " + output[9]);
        

    }

}
