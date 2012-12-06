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
import java.io.FileWriter;
import java.io.IOException;
import java.util.Locale;
import java.util.Scanner;

import org.encog.ml.data.MLData;
import org.encog.ml.data.MLDataPair;
import org.encog.ml.train.MLTrain;
import org.encog.neural.NeuralNetworkError;
import org.encog.neural.networks.BasicNetwork;
import org.encog.persist.EncogDirectoryPersistence;

/**
 * Abstract class that models the neural networks used on
 * trainings by the ILPNetworkTraining project and Android
 * application IntelligentLockPattern.
 * 
 * Provides the basics operations and attributes presented on a network as
 * number of inputs units, number of output units, number of hidden layers 
 * and units on each layer, permited max error on the training, 
 * max number of iterations and utilized training method.
 * 
 * Presents methods for saving (save) and loading (load) the network 
 * trained on a text file.
 * 
 * Uses the Encog Framework {http://www.heatonresearch.com/encog} 
 * to build the differents configurations of the networks.
 * The BasicNetwork class is used as the base structure
 * for all the network used on the job.
 * 
 * The permited max error on the training default is 0.1.
 * 
 * @author Lucas André de Alencar
 *
 */
public abstract class AbstractNeuralNetwork {
	
	protected static final double MAX_ERROR = 0.1;
	
	protected double max_error;
	protected int max_iterations;
	
	protected int numInputs;
	protected int numOutputs;
	protected int [] hiddenLayers;
	
	protected BasicNetwork network;
	protected MLTrain training;
	
	protected double trainError;
	
	/**
	 * Constructor used on the construction of a new network structure.
	 * 
	 * The hidden layers should be specified as an array where each position
	 * represents a new layer and the number contained on the position represents
	 * the number of units on the layer.
	 * 
	 * Ex: [5, 3] => 2 hidden layers. The first one has 5 units and the second one has 3 units.
	 * 
	 * @param numInputs Number of input units
	 * @param hiddenLayers Array with the number of units on each hidden layer
	 * @param numOutputs Number of output units
	 */
	public AbstractNeuralNetwork(int numInputs, int[] hiddenLayers, int numOutputs) {
		this.numInputs = numInputs;
		this.hiddenLayers = hiddenLayers;
		this.numOutputs = numOutputs;
	}
	
	/**
	 * Constructor that loads the infos on the network from a file.
	 * 
	 * @param file File used to load the network info
	 * @throws FileNotFoundException
	 */
	public AbstractNeuralNetwork(File file) throws FileNotFoundException {
		load(file);
		numInputs = network.getInputCount();
		numOutputs = network.getOutputCount();
		hiddenLayers = new int[network.getLayerCount() - 2];
		for (int i = 1; i < network.getLayerCount() - 1; i++) {
			hiddenLayers[i - 1] = network.getLayerNeuronCount(i);
		}
	}
	
	public int getNumInputs() {
		return numInputs;
	}
	
	public int getNumOutputs() {
		return numOutputs;
	}
	
	public int[] getHiddenLayers() {
		return hiddenLayers;
	}
	
	public BasicNetwork getNetwork() {
		return network;
	}
	
	public void setTrainStrategy(MLTrain train) {
		this.training = train;
	}
	
	public MLTrain getTrainStrategy() {
		return training;
	}
	
	/**
	 * Inserts the input values on the network and computes the results
	 * returning an array of values from all the output layer's units.
	 * 
	 * @param input Values inserted on network input
	 * @return Network's computation resulted from the input values
	 */
	public MLData compute(MLDataPair input) {
		return network.compute(input.getInput());
	}
	
	/**
	 * Inserts the input values on the network and computes the results
	 * returning an array of values from all the output layer's units.
	 * 
	 * @param input Values inserted on network input
	 * @return Network's computation resulted from the input values
	 */
	public double[] compute(double [] input) {
		if (input.length != numInputs) throw new IllegalStateException("Input size isn't match with Number of Inputs defined.");
		double [] output = new double[input.length];
		network.compute(input, output);
		return output;
	}
	
	public void setMaxError(double max_error) {
		if (max_error > 0)
			this.max_error = max_error;
		else
			throw new NeuralNetworkError("Max error must be greater than 0.");
	}
	
	public double getMaxError() {
		if (max_error != 0)
			return max_error;
		else
			return MAX_ERROR;
	}

	public void setMaxIterations(int max_iterations) {
		this.max_iterations = max_iterations;
	}
	
	public int getMaxIterations() {
		return this.max_iterations;
	}
	
	/**
	 * Calculates the MSE (Mean Squared Error) based
	 * expected ideal and the resulted output.
	 * 
	 * @param ideal Values expected as output
	 * @param output Values resluted as output
	 * @return double MSE
	 */
	public double calculateError(MLData ideal, MLData output) {
		double sum = 0, delta = 0;
		for (int i = 0; i < ideal.size(); i++) {
			delta = ideal.getData(i) - output.getData(i);
			sum += (delta * delta);
		}
		return sum / (double) ideal.size();
	}
	
	/**
	 * Calculates the MSE (Mean Squared Error) based
	 * expected ideal and the resulted output.
	 * 
	 * @param ideal Values expected as output
	 * @param output Values resluted as output
	 * @return double MSE
	 */
	public double calculateError(double [] ideal, double [] output) {
		double sum = 0, delta = 0;
		for (int i = 0; i < ideal.length; i++) {
			delta = ideal[i] - output[i];
			sum += (delta * delta);
		}
		return sum / (double) ideal.length;
	}
	
	public double getTrainError() {
		return trainError;
	}
	
	public void updateTrainError() {
		trainError = training.getError();
	}
	
	public void inspect() {
		System.out.println("NumInputs = " + numInputs);
		System.out.println("Hidden layers = " + hiddenLayers[0] + "-" + hiddenLayers[1] + "-" + hiddenLayers[2]);
		System.out.println("NumOutputs = " + numOutputs);
		System.out.println("MaxError = " + this.getMaxError());
		System.out.println("MaxIterations = " + this.getMaxIterations());
		
		if (trainError > 0)
			System.out.println("TrainError = " + this.trainError);
		
		if (this.getTrainStrategy() != null)
			System.out.println("Training = " + this.getTrainStrategy().getClass().getName());
	}
	
	/**
	 * Saves the info contained on the class on text files.
	 * The network is saved on file_name.network and the
	 * training error is saved on file_name.training.
	 * 
	 * @param file
	 * @throws IOException
	 */
	public void save(File file) throws IOException {
		File networkFile = new File(file.getAbsolutePath() + ".network");
		EncogDirectoryPersistence.saveObject(networkFile, network);
		
		File trainingFile = new File(file.getAbsolutePath() + ".training");
		FileWriter writer = new FileWriter(trainingFile);
		writer.write(String.format(Locale.US, "%.20f", getTrainError()));
		writer.close();
	}
	
	/**
	 * Loads the info stored in files on the class.
	 * The network is loaded from file_name.network and the
	 * training error is loaded from file_name.training.
	 * 
	 * @param file
	 * @throws FileNotFoundException
	 */
	private void load(File file) throws FileNotFoundException {
		File networkFile = new File(file.getAbsolutePath() + ".network");
		this.network = (BasicNetwork) EncogDirectoryPersistence.loadObject(networkFile);
		
		File trainingFile = new File(file.getAbsolutePath() + ".training");
		Scanner scanner = new Scanner(trainingFile);
		this.trainError = new Double(scanner.next());
		scanner.close();
	}
}
