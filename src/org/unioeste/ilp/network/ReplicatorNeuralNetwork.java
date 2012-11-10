package org.unioeste.ilp.network;


import java.io.File;

import org.encog.engine.network.activation.ActivationFunction;
import org.encog.engine.network.activation.ActivationRamp;
import org.encog.engine.network.activation.ActivationSigmoid;
import org.encog.neural.networks.BasicNetwork;
import org.encog.neural.networks.layers.BasicLayer;

public class ReplicatorNeuralNetwork extends AbstractNeuralNetwork {
	
	public static final int DEFAULT_NUM_HIDDEN_LAYERS = 3;
	
	public ReplicatorNeuralNetwork(int numInputs, int [] hiddenLayers) {
		super(numInputs, hiddenLayers, numInputs);
		construct(numInputs, hiddenLayers, new ActivationSigmoid(), true);
	}
	
	public ReplicatorNeuralNetwork(File file) {
		super(file);
	}
	
	private void construct(int numInputs, int [] hiddenLayers, ActivationFunction activationFunction, boolean hasBias) {
		network = new BasicNetwork();
		network.addLayer(new BasicLayer(activationFunction, hasBias, numInputs)); // Input Layer
		constructHiddenLayers(hiddenLayers); // Hidden layers
		network.addLayer(new BasicLayer(activationFunction, hasBias, numInputs)); // Output Layer
		network.getStructure().finalizeStructure();
		network.reset();
	}
	
	private void constructHiddenLayers(int [] hiddenLayers) {
		if (hiddenLayers.length != DEFAULT_NUM_HIDDEN_LAYERS)
			throw new IllegalStateException("Wrong number of hidden layers. Hidden layer must have length " + DEFAULT_NUM_HIDDEN_LAYERS + ".");
		
		for (int i = 0; i < hiddenLayers.length; i++) {
			if (hiddenLayers[i] <= 0)
				throw new IllegalStateException("Hidden layer " + (i + 2) + " must be greater than 0.");
		}
		
		network.addLayer(new BasicLayer(new ActivationSigmoid(), true, hiddenLayers[0])); // Hidden layer 2
		network.addLayer(new BasicLayer(new ActivationRamp(), true, hiddenLayers[1])); // Hidden layer 3
		network.addLayer(new BasicLayer(new ActivationSigmoid(), true, hiddenLayers[2])); // Hidden layer 4
	}
	
	public double getTrainError() {
		return training.getError();
	}
}
