

public class Neuron {
    public double[] weights;
    public double[] inputs;
    private double bias;
    private double output;
    
    private double[] weightGradients;
    private double biasGradient;

    

    public Neuron(int inputSize) {
        weightGradients = new double[inputSize];
        weights = new double[inputSize];
        
        for(int i = 0; i < inputSize; i++) {
            weights[i] = (Math.random() - 0.5) * 0.1; 
        }
        
        bias = (Math.random() - 0.5) * 0.1;
    }
    
    public double activate(double[] inputs) {
        this.inputs = inputs;
        
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
    
    public double sigmoidDerivative(double x) {
        return x * (1 - x);
    }
    
    public double getOutput() {
        return output;
    }
    
    public double[] getInput() {
        return this.inputs;
    }
    
    public double[] getWeights() {
        return this.weights;
    }
    
    public void updateWeights(double learningRate,double delta) {
        for(int i = 0; i < weights.length; i++) {
            // change in weight = learning rate * δj * Oj
            weights[i] += learningRate * delta * inputs[i];
        }
        
        bias += learningRate * delta;
    }
}
