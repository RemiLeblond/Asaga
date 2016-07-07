package algorithms.sequential

import java.lang.Math.pow

import algorithms.StochasticGradientDescent
import models.{LeastSquares, LogisticRegression}
import utils._

import scala.util.Random._

/**
  * Created by rleblond on 1/14/16.
  */

/**
  * Lagged updates implementation of Saga to take advantage of data sparsity (see http://arxiv.org/abs/1309.2388)
  * See StochasticGradientDescent for parameter explanation.
  */
abstract class LaggedUpdatesSaga(inputs: CSRMatrix,
                                 output: Array[Double],
                                 stepSize: Double,
                                 iterationsFactor: Int,
                                 historyRatio: Int,
                                 lambda: Double)
  extends StochasticGradientDescent(inputs, output, stepSize, iterationsFactor, historyRatio, lambda) {

  val lags = Array.fill[Int](dimension)(0)
  val lagsHistory = Array.fill[Int](iterations / historyRatio + 1, dimension)(0)
  val gradientHistory = Array.fill[Double](iterations / historyRatio + 1, dimension)(0d)

  def run(printLosses: Boolean, verbose: Boolean) {
    // initialization
    initialTimeStamp = System.currentTimeMillis()
    parametersHistory(0) = (0, Array.fill[Double](dimension)(initialParameterValue), 0)
    val (currentAverageGradient, historicalGradients) = computeFullGradient(inputs, output, parameters)
    val scaling = 1.0 - stepSize * lambda

    // cummulative lag
    val cumscaling = Array.fill[Double](historyRatio + 1)(0.0)
    for (i <- 1 until cumscaling.size) {
      cumscaling(i) = cumscaling(i-1) + pow(scaling, i)
    }

    for (i <- 1 to iterations + 1) {
      // pick a random number
      val index = nextInt(n)

      // do the lagged updates
      for (j <- inputs.indPtr(index) until inputs.indPtr(index + 1)) {
        val k = inputs.indices(j)
        val v = inputs.data(j)
        parameters(k) = pow(scaling, i - lags(k)) * parameters(k) - stepSize * (cumscaling(i - lags(k)) * currentAverageGradient(k))
        lags(k) = i
      }

      // compute a partial gradient
      val scalar = computePartialGradient(inputs, output(index), parameters, index)
      val oldScalar = historicalGradients(index)

      // update
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

      // store for later suboptimality computation
      if (i % historyRatio == 0) {
        // do a full lagged updates to keep cumscaling lower than historyRatio
        // this way (although not as efficient) we avoid the need to do
        // the re-scaling at the end
        for (k <- parameters.indices) {
          parameters(k) = pow(scaling, i - lags(k)) * parameters(k) -
            stepSize * (cumscaling(i - lags(k)) * currentAverageGradient(k))
          lags(k) = i
        }

        lagsHistory(i / historyRatio) = (0 until lags.size).map(lags(_)).toArray
        parametersHistory(i / historyRatio) = (System.currentTimeMillis() - initialTimeStamp, (0 until parameters.size).map(parameters(_)).toArray, 0)
        gradientHistory(i / historyRatio) = (0 until currentAverageGradient.size).map(currentAverageGradient(_)).toArray
      }
    }

    writeLosses(printLosses, verbose)
  }
}


class LaggedSagaLS(inputs: CSRMatrix,
                   output: Array[Double],
                   stepSize: Double,
                   iterations_factor: Int,
                   historyRatio: Int,
                   lambda: Double)
  extends LaggedUpdatesSaga(inputs, output, stepSize, iterations_factor, historyRatio, lambda) with LeastSquares


class LaggedSagaLogistic(inputs: CSRMatrix,
                         output: Array[Double],
                         stepSize: Double,
                         iterations_factor: Int,
                         historyRatio: Int,
                         lambda: Double)
  extends LaggedUpdatesSaga(inputs, output, stepSize, iterations_factor, historyRatio, lambda) with LogisticRegression


/**
  * Entry point to launch the method with the compiled JAR.
  */
object LaggedUpdatesSaga {
  def main(args: Array[String]) {
    if (args.size != 6) {
      println("Usage: LaggedUpdateSaga <data><model><stepsize><iterations><history><lambda>")
      System.exit(0)
    }
    val (inputs, output) = args(0) match {
      case "rcv1" => loadDataRCV1()
    }
    val (stepSize, iterationsFactor, historyRatio, lambda) = (args(2).toDouble, args(3).toInt, args(4).toInt, args(5).toInt)

    val lusls = args(1) match {
      case "LS" => new LaggedSagaLS(inputs, output, stepSize, iterationsFactor, historyRatio, lambda)
      case "Logistic" => new LaggedSagaLogistic(inputs, output, stepSize, iterationsFactor, historyRatio, lambda)
    }

    lusls.run(true, false)
  }
}