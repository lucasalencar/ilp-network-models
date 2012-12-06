/*
 * "Copyright 2012 Lucas André de Alencar"
 * 
 * This file is part of ILPNetworkModels.
 * 
 * ILPNetworkModels is free software: you can redistribute it and/or modify 
 * it under the terms of the GNU General Public License as published by 
 * the Free Software Foundation, either version 3 of the License, or 
 * (at your option) any later version.
 * 
 * ILPNetworkModels is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of 
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the 
 * GNU General Public License for more details. 
 * 
 * You should have received a copy of the GNU General Public License 
 * along with ILPNetworkModels.  If not, see <http://www.gnu.org/licenses/>.
 */

package org.unioeste.ilp.network;

import java.io.File;
import java.io.FileNotFoundException;

import org.encog.engine.network.activation.ActivationFunction;
import org.encog.engine.network.activation.ActivationRamp;
import org.encog.engine.network.activation.ActivationSigmoid;
import org.encog.neural.networks.BasicNetwork;
import org.encog.neural.networks.layers.BasicLayer;

/**
 * Class that molds a replicator neural network that must have
 * on the output layer the same number of units presented on the
 * input layer, and the results got by the computation must be
 * the most proximate from the input data.
 * 
 * Requires that the network have 3 hidden layers on total.
 * 
 * The default activation function on all the neurons, except 
 * the ones on the middle layer, is the sigmoid. The activation on
 * the middle layer is the ramp activation.
 * 
 * Constructing the network:
 * 
 * The hidden layers should be specified as an array where each position
 * represents a new layer and the number contained on the position represents
 * the number of units on the layer.
 * 
 * Ex: [5, 3] => 2 hidden layers. The first one has 5 units and the second one has 3 units.
 * 
 * @author Lucas André de Alencar
 *
 */
public class ReplicatorNeuralNetwork extends AbstractNeuralNetwork {
	
	public static final int DEFAULT_NUM_HIDDEN_LAYERS = 3;
	
	/**
	 * Constructor for new replicator neural networks.
	 * 
	 * @param numInputs Number of inputs units
	 * @param hiddenLayers Array with the number of units on each hidden layer
	 */
	public ReplicatorNeuralNetwork(int numInputs, int [] hiddenLayers) {
		super(numInputs, hiddenLayers, numInputs);
		construct(numInputs, hiddenLayers, new ActivationSigmoid(), true);
	}
	
	/**
	 * Constructor that loads the infos on the network from a file.
	 * 
	 * @param file File used to load the network info
	 * @throws FileNotFoundException
	 */
	public ReplicatorNeuralNetwork(File file) throws FileNotFoundException {
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
	
	// Values used on the normalization process. Here is specified the input and output min and max values expected.
	// Only valid for inputs that have 17 input units.
	public static final double [] input_low_norm = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
	public static final double [] input_high_norm = {1, 1, 5000, 1, 1, 5000, 1, 1, 5000, 1, 1, 5000, 1, 1, 5000, 1, 1};
	public static final double [] output_low_norm = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
	public static final double [] output_high_norm = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
}
