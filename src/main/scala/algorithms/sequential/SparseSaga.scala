package algorithms.sequential

import algorithms.StochasticGradientDescent
import models.{LeastSquares, LogisticRegression}
import utils._

import scala.util.Random._

/**
  * Created by rleblond on 1/14/16.
  */

/**
  * Implementation of our Sparse Saga algorithm (see https://arxiv.org/abs/1606.04809)
  * See StochasticGradientDescent for parameter explanation.
  */
abstract class SparseSaga(inputs: CSRMatrix,
                          output: Array[Double],
                          stepSize: Double,
                          iterationsFactor: Int,
                          historyRatio: Int,
                          lambda: Double)
  extends StochasticGradientDescent(inputs, output, stepSize, iterationsFactor, historyRatio, lambda) {

  // compute the pi probabilities
  val inversePi = computeInversePis(inputs, dimension)

  def run(printLosses: Boolean, verbose: Boolean) {
    // initialization
    initialTimeStamp = System.currentTimeMillis()
    parametersHistory(0) = (0, Array.fill[Double](dimension)(initialParameterValue), 0)
    val (currentAverageGradient, historicalGradients) = computeFullGradient(inputs, output, parameters)

    for (i <- 1 to iterations + 1) {
      // pick a random number
      val index = nextInt(n)

      // compute a partial gradient
      val scalar = computePartialGradient(inputs, output(index), parameters, index)
      val oldScalar = historicalGradients(index)

      // update
      for (j <- inputs.indPtr(index) until inputs.indPtr(index + 1)) {
        val k = inputs.indices(j)
        val v = inputs.data(j)
        parameters(k) -= stepSize * (v * (scalar - oldScalar) + inversePi(k) * (currentAverageGradient(k) + lambda * parameters(k)))
      }

      // store parameters for later suboptimality computation
      if (i % historyRatio == 0) {
        parametersHistory(i / historyRatio) = (System.currentTimeMillis() - initialTimeStamp, (0 until parameters.size).map(parameters(_)).toArray, 0) // this is needed to get a copy of the parameter array and not a reference to said array which keeps changing
      }

      // update the average gradient and the historical one
      historicalGradients(index) = scalar
      for (j <- inputs.indPtr(index) until inputs.indPtr(index + 1)) {
        val k = inputs.indices(j)
        val v = inputs.data(j)
        currentAverageGradient(k) += (scalar - oldScalar) * v / n
      }
    }

    writeLosses(printLosses, verbose)
  }
}


class SparseSagaLS(inputs: CSRMatrix,
                   output: Array[Double],
                   stepSize: Double,
                   iterations_factor: Int,
                   historyRatio: Int,
                   lambda: Double)
  extends SparseSaga(inputs, output, stepSize, iterations_factor, historyRatio, lambda: Double) with LeastSquares


class SparseSagaLogistic(inputs: CSRMatrix,
                         output: Array[Double],
                         stepSize: Double,
                         iterations_factor: Int,
                         historyRatio: Int,
                         lambda: Double)
  extends SparseSaga(inputs, output, stepSize, iterations_factor, historyRatio, lambda: Double) with LogisticRegression


/**
  * Entry point to launch the method with the compiled JAR.
  */
object SparseSaga {
  def main(args: Array[String]) {
    if (args.size != 6) {
      println("Usage: SparseSaga <data><model><stepsize><iterations><history><lambda>")
      System.exit(0)
    }
    val (inputs, output) = args(0) match {
      case "rcv1" => loadDataRCV1()
    }
    val (stepSize, iterationsFactor, historyRatio, lambda) = (args(2).toDouble, args(3).toInt, args(4).toInt, args(5).toInt)

    val sls = args(1) match {
      case "LS" => new SparseSagaLS(inputs, output, stepSize, iterationsFactor, historyRatio, lambda)
      case "Logistic" => new SparseSagaLogistic(inputs, output, stepSize, iterationsFactor, historyRatio, lambda)
    }

    sls.run(true, false)
  }
}