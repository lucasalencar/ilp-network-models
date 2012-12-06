Intelligent Lock Pattern Project
================================

This project is part of a bigger project named Intelligent Lock Pattern that has the goal of build a behavioral biometric system on Android based on the lock patterns authentication system present on the OS.

The project started as a university course conclusion work in 2012 and now it's open for new ideas and improvements.

The basic idea is make an artificial intelligence that learns how the user draws his pattern on the screen. Using the knowledge learned, the system verifies if the user that is drawing the pattern is authentic through the inserted characteristics during the drawing.

ILP Network Models
------------------

**ILP Network Models** holds all the neural network models used to learn the characteristics from the user.

Includes the [Encog Framework](http://www.heatonresearch.com/encog) on the libs folder that models the networks used on the project.

The others sub projects:
*	[ILP Models Generator](https://github.com/lucasandre/ilp-models-generator)
*	[ILP Network Training](https://github.com/lucasandre/ilp-network-training)
*	[Intelligent Lock Pattern](https://github.com/lucasandre/intelligent-lock-pattern)