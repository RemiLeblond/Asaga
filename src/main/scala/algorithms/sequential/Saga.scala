package algorithms.sequential

import algorithms.StochasticGradientDescent
import models.{LeastSquares, LogisticRegression}
import utils._

import scala.util.Random._

/**
 * Created by rleblond on 1/14/16.
 */

/**
  * Classic Saga algorithm for generalized linear models.
  * See StochasticGradientDescent for parameter explanation.
  */
abstract class Saga(inputs: CSRMatrix,
                    output: Array[Double],
                    stepSize: Double,
                    iterationsFactor: Int,
                    historyRatio: Int,
                    lambda: Double)
  extends StochasticGradientDescent(inputs, output, stepSize, iterationsFactor, historyRatio, lambda) {

  def run(printLosses: Boolean, verbose: Boolean) {
    // initialization
    initialTimeStamp = System.currentTimeMillis()
    val dimension = inputs.nCols
    parametersHistory(0) = (0, (1 to dimension).map(_ => initialParameterValue).toArray, 0)
    val (currentAverageGradient, historicalGradients) = computeFullGradient(inputs, output, parameters)

    for (i <- 1 to iterations + 1) {
      // pick a random number
      val index = nextInt(n)

      // compute a partial gradient
      val scalar = computePartialGradient(inputs, output(index), parameters, index)
      val oldScalar = historicalGradients(index)

      // take a step
      for (u <- parameters.indices) {
        parameters(u) -= stepSize * (currentAverageGradient(u) + lambda * parameters(u))
      }
      for (j <- inputs.indPtr(index) until inputs.indPtr(index + 1)) {
        val k = inputs.indices(j)
        val v = inputs.data(j)
        parameters(k) -= stepSize * (scalar - oldScalar) * v
      }

      // update the average gradient and the historical one
      historicalGradients(index) = scalar
      for (j <- inputs.indPtr(index) until inputs.indPtr(index + 1)) {
        val k = inputs.indices(j)
        val v = inputs.data(j)
        currentAverageGradient(k) += (scalar - oldScalar) * v / n
      }
      // store parameters for later suboptimality computation
      if (i % historyRatio == 0) {
        parametersHistory(i / historyRatio) = (System.currentTimeMillis() - initialTimeStamp, (0 until parameters.size).map(parameters(_)).toArray, 0) // this is needed to get a copy of the parameter array and not a reference to said array which keeps changing
      }
    }

    writeLosses(printLosses, verbose)
  }
}


class SagaLS(inputs: CSRMatrix,
             output: Array[Double],
             stepSize: Double,
             iterations_factor: Int,
             historyRatio: Int,
             lambda: Double)
  extends Saga(inputs, output, stepSize, iterations_factor, historyRatio, lambda) with LeastSquares


class SagaLogistic(inputs: CSRMatrix,
                   output: Array[Double],
                   stepSize: Double,
                   iterations_factor: Int,
                   historyRatio: Int,
                   lambda: Double)
  extends Saga(inputs, output, stepSize, iterations_factor, historyRatio, lambda) with LogisticRegression


/**
  * Entry point to launch the method with the compiled JAR
  */
object Saga {
  def main(args: Array[String]) {
    if (args.size != 6) {
      println("Usage: Saga <data><model><stepsize><iterations><history><lambda>")
      System.exit(0)
    }
    val (inputs, output) = args(0) match {
      case "rcv1" => loadDataRCV1()
    }
    val (stepSize, iterationsFactor, historyRatio, lambda) = (args(2).toDouble, args(3).toInt, args(4).toInt, args(5).toDouble)

    val sls = args(1) match {
      case "LS" => new SagaLS(inputs, output, stepSize, iterationsFactor, historyRatio, lambda)
      case "Logistic" => new SagaLogistic(inputs, output, stepSize, iterationsFactor, historyRatio, lambda)
    }

    sls.run(true, false)
  }
}
