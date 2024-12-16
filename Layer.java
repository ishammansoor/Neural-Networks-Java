
public class Layer {
    private Neuron[] layer;
    

    public Layer(int numNeurons, int inputSize) {
        layer = new Neuron[numNeurons];
        
        for(int i = 0; i < numNeurons; i++) {
            layer[i] = new Neuron(inputSize);
        }
    }
    
    public double[] forward(double[] inputs) {
        double[] outputs = new double[layer.length];
        
        for(int i = 0; i < layer.length; i++) {
            outputs[i] = layer[i].activate(inputs);
        }
        
        return outputs;
    }

}
