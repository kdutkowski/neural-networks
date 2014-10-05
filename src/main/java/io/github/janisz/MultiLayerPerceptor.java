package io.github.janisz;

import org.encog.Encog;
import org.encog.engine.network.activation.ActivationSigmoid;
import org.encog.ml.data.MLData;
import org.encog.ml.data.MLDataPair;
import org.encog.ml.data.MLDataSet;
import org.encog.ml.data.basic.BasicMLDataSet;
import org.encog.neural.networks.BasicNetwork;
import org.encog.neural.networks.layers.BasicLayer;
import org.encog.neural.networks.training.propagation.resilient.ResilientPropagation;

public class MultiLayerPerceptor {

  public static double XOR_INPUT[][] = {
      {0.0, 0.0}, {1.0, 0.0},
      {0.0, 1.0}, {1.0, 1.0}
  };

  public static double XOR_IDEAL[][] = {{0.0}, {1.0}, {1.0}, {0.0}};

  public String buildSampleXorNetwork() {
    // create a neural network, without using a factory
    BasicNetwork network = new BasicNetwork();
    network.addLayer(new BasicLayer(null, true, 2));
    network.addLayer(new BasicLayer(new ActivationSigmoid(), true, 3));
    network.addLayer(new BasicLayer(new ActivationSigmoid(), false, 1));
    network.getStructure().finalizeStructure();
    network.reset();

    // create training data
    MLDataSet trainingSet = new BasicMLDataSet(XOR_INPUT, XOR_IDEAL);

    // train the neural network
    final ResilientPropagation train = new ResilientPropagation(network, trainingSet);

    int epoch = 1;

    do {
      train.iteration();
      System.out.println("Epoch #" + epoch + " Error:" + train.getError());
      epoch++;
    } while (train.getError() > 0.01);
    train.finishTraining();

    // test the neural network
    StringBuilder stringBuilder = new StringBuilder("Neural Network Results:\n");
    for (MLDataPair pair : trainingSet) {
      final MLData output = network.compute(pair.getInput());
      stringBuilder
          .append(pair.getInput().getData(0))
          .append(",")
          .append(pair.getInput().getData(1))
          .append(", actual=")
          .append(output.getData(0))
          .append(",ideal=")
          .append(pair.getIdeal().getData(0))
          .append('\n');
    }

    Encog.getInstance().shutdown();
    return stringBuilder.toString();
  }
}
