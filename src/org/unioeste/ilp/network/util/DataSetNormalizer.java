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

package org.unioeste.ilp.network.util;

/**
 * Class responsible for the data normalization, 
 * used before any training or tests of networks.
 * 
 * @author Lucas André de Alencar
 *
 */
public class DataSetNormalizer {
	public static double normalize(double input_low, double input_high, double output_low, double output_high, double value) {
		return ((value - input_low) / (input_high - input_low) * (output_high - output_low) + output_low);
	}
	
	public static double deNormalize(double input_low, double input_high, double output_low, double output_high, double value) {
		return ((input_low - input_high) * value - output_high * input_low + input_high * output_low) / (output_low - output_high);
	}
	
	public static double[] normalize(double input_low, double input_high, double output_low, double output_high, double[] values) {
		double[] normalized = new double[values.length];
		for (int i = 0; i < values.length; i++) {
			normalized[i] = normalize(input_low, input_high, output_low, output_high, values[i]);
		}
		return normalized;
	}
	
	public static double[] deNormalize(double input_low, double input_high, double output_low, double output_high, double[] values) {
		double[] deNormalized = new double[values.length];
		for (int i = 0; i < values.length; i++) {
			deNormalized[i] = deNormalize(input_low, input_high, output_low, output_high, values[i]);
		}
		return deNormalized;
	}
	
	public static double[][] normalize(double[] column_input_low, double[] column_input_high, 
			double[] column_output_low, double[] column_output_high, double[][] values) {
		
		double[][] normalized = new double[values.length][values[0].length];
		for (int i = 0; i < values.length; i++) {
			for (int j = 0; j < values[0].length; j++) {
				normalized[i][j] = normalize(
						column_input_low[j], column_input_high[j], 
						column_output_low[j], column_output_high[j], 
						values[i][j]
				);
			}
		}
		return normalized;
	}
	
	public static double[] normalize(double[] column_input_low, double[] column_input_high,
			double[] column_output_low, double[] column_output_high, double[] values) {
		
		double[] normalized = new double[values.length];
		for (int j = 0; j < values.length; j++) {
			normalized[j] = normalize(
					column_input_low[j], column_input_high[j], 
					column_output_low[j], column_output_high[j], 
					values[j]
			);
		}
		return normalized;
	}
}