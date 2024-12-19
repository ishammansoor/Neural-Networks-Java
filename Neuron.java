

public class Neuron {
    private double[] weights;
    private double bias;
    private double output;
    

    public Neuron(int inputSize) {
        weights = new double[inputSize];
        
        for(int i = 0; i < inputSize; i++) {
            weights[i] = Math.random();
        }
        
        bias = Math.random() * 0.1;
    }
    
    public double activate(double[] inputs) {
        double sum = bias;
        
        for(int i = 0; i < weights.length; i++) {
            sum += inputs[i] * weights[i];
        }
        
        output = sigmoid(sum);
        
        return output;
    }
    
    public double sigmoid(double x) {
        return 1 / (1 + Math.exp(-x));
    }
    
    
    public double getOutput() {
        return output;
    }

}
