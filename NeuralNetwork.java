
public class NeuralNetwork {
    
    private Layer hiddenLayer;
    private Layer outputLayer;
    
    public NeuralNetwork() {
        hiddenLayer = new Layer(2, 2); // 2 neurons in the hidden layer, 2 inputs
        outputLayer = new Layer(1, 2); // 1 neuron in the output layer, 2 inputs from hidden layer
    }
    
    public double[] forward(double[] inputs) {
        double[] hiddenOutputs = hiddenLayer.forward(inputs);
        return outputLayer.forward(hiddenOutputs);
    }


    public static void main(String[] args) {
        
        NeuralNetwork nn = new NeuralNetwork();
        
        double[] inputs = {0.5, 0.8};
        double[] output = nn.forward(inputs);
        
        System.out.println("Predicted Output: " + output[0]);
        

    }

}
