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
	
	public AbstractNeuralNetwork(int numInputs, int[] hiddenLayers, int numOutputs) {
		this.numInputs = numInputs;
		this.hiddenLayers = hiddenLayers;
		this.numOutputs = numOutputs;
	}
	
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
	
	public MLData compute(MLDataPair input) {
		return network.compute(input.getInput());
	}
	
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
	
	public double calculateError(MLData ideal, MLData output) {
		double sum = 0, delta = 0;
		for (int i = 0; i < ideal.size(); i++) {
			delta = ideal.getData(i) - output.getData(i);
			sum += (delta * delta);
		}
		return sum / (double) ideal.size();
	}
	
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
	
	public void save(File file) throws IOException {
		File networkFile = new File(file.getAbsolutePath() + ".network");
		EncogDirectoryPersistence.saveObject(networkFile, network);
		
		File trainFile = new File(file.getAbsolutePath() + ".train");
		FileWriter writer = new FileWriter(trainFile);
		writer.write(String.format(Locale.US, "%.20f", getTrainError()));
		writer.close();
	}
	
	private void load(File file) throws FileNotFoundException {
		File networkFile = new File(file.getAbsolutePath() + ".network");
		this.network = (BasicNetwork) EncogDirectoryPersistence.loadObject(networkFile);
		
		File trainFile = new File(file.getAbsolutePath() + ".train");
		Scanner scanner = new Scanner(trainFile);
		this.trainError = new Double(scanner.next());
		scanner.close();
	}
}
