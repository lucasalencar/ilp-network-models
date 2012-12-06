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

import org.encog.engine.network.activation.ActivationFunction;
import org.encog.engine.network.activation.ActivationSigmoid;
import org.encog.neural.networks.BasicNetwork;
import org.encog.neural.networks.layers.BasicLayer;

/**
 * Class inherited from AbstractNeuralNetwork that lets build 
 * basic neural network structures. Letting choose the number 
 * of units and layers, activation function and presence of a 
 * bias unit on each layer.
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
public class NeuralNetwork extends AbstractNeuralNetwork {
	
	/**
	 * Constructor that builds a basic network setting
	 * as default the sigmoid activation function and the
	 * enables the bias unit on the layers.
	 * 
	 * @param numInputs Number of input units
	 * @param hiddenLayers Array with the number of units on each hidden layer
	 * @param numOutputs Number of output units
	 */
	public NeuralNetwork(int numInputs, int [] hiddenLayers, int numOutputs) {
		this(numInputs, hiddenLayers, numOutputs, new ActivationSigmoid(), true);
	}
	
	/**
	 * Constructor where is possible to specify the
	 * activation function on all the layers and presence
	 * of bias units on each layer.
	 * 
	 * @param numInputs Number of input units
	 * @param hiddenLayers Array with the number of units on each hidden layer
	 * @param numOutputs Number of output units
	 * @param activation Activation function that is applied on every neuron on the network
	 * @param hasBias Flag that sets the presence of bias units on the layers
	 */
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
