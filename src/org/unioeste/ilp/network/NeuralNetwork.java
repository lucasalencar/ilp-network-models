package org.unioeste.ilp.network;


import org.encog.engine.network.activation.ActivationFunction;
import org.encog.engine.network.activation.ActivationSigmoid;
import org.encog.neural.networks.BasicNetwork;
import org.encog.neural.networks.layers.BasicLayer;

public class NeuralNetwork extends AbstractNeuralNetwork {
	
	public NeuralNetwork(int numInputs, int [] hiddenLayers, int numOutputs) {
		this(numInputs, hiddenLayers, numOutputs, new ActivationSigmoid(), true);
	}
	
	public NeuralNetwork(int numInputs, int [] hiddenLayers, int numOutputs, ActivationFunction activation, boolean hasBias) {
		super(numInputs, hiddenLayers, numOutputs);
		construct(numInputs, hiddenLayers, numOutputs, activation, hasBias);
	}
	
	private void construct(int numInputs, int [] hiddenLayers, int numOutputs, ActivationFunction activation, boolean hasBias) {
		network = new BasicNetwork();
		network.addLayer(new BasicLayer(activation, hasBias, numInputs)); // Input layer
		
		if (hiddenLayers != null) {
			// Hidden layers
			for (int i = 0; i < hiddenLayers.length; i++) {
				network.addLayer(new BasicLayer(activation, hasBias, hiddenLayers[i]));
			}
		}
		
		network.addLayer(new BasicLayer(activation, hasBias, numOutputs)); // Output layer
		network.getStructure().finalizeStructure();
		network.reset();
	}	
}
