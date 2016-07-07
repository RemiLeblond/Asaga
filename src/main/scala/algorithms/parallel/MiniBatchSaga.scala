package algorithms.parallel

import algorithms.StochasticGradientDescent
import models.{LeastSquares, LogisticRegression}
import utils._

import scala.concurrent.ExecutionContext.Implicits.global
import scala.concurrent.duration._
import scala.concurrent.{Await, Future}
import scala.util.Random._


/**
  * Mini-batch, sparse Saga
  * For parameter details, see StochasticGradientDescent.
  */
abstract class MiniBatchSaga(inputs: CSRMatrix,
                             output: Array[Double],
                             stepSize: Double,
                             iterationsFactor: Int,
                             historyRatio: Int,
                             cores: Int,
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
      //launch cores gradient computation in parallel
      val futures = (1 to cores).map(
        i => Future(singleStep(historicalGradients))
      )
      val singleSteps = Await.result(Future.sequence(futures), Duration.Inf)

      for ((index, scalar, oldScalar) <- singleSteps;
           j <- inputs.indPtr(index) until inputs.indPtr(index + 1)) {
        val k = inputs.indices(j)
        val v = inputs.data(j)
        parameters(k) -= stepSize * (v * (scalar - oldScalar) + inversePi(k) * (currentAverageGradient(k) + lambda * parameters(k))) / cores
      }

      // update the average gradient and the historical one
      for ((index, scalar, oldScalar) <- singleSteps) {
        for (j <- inputs.indPtr(index) until inputs.indPtr(index + 1)) {
          val k = inputs.indices(j)
          val v = inputs.data(j)
          currentAverageGradient(k) += (scalar - oldScalar) * v / n
        }

        historicalGradients(index) = scalar
      }

      // store parameters for later suboptimality computation
      if (i % historyRatio == 0) {
        parametersHistory(i / historyRatio) = (System.currentTimeMillis() - initialTimeStamp, (0 until parameters.size).map(parameters(_)).toArray, 0) // this is needed to get a copy of the parameter array and not a reference to said array which keeps changing
      }
    }

    // compute the losses and print them
    writeLosses(printLosses, verbose)
  }

  def singleStep(historicalGradients: Array[Double]): (Int, Double, Double) = {
    // pick a random number
    val index = nextInt(n)

    // compute a partial gradient
    val scalar = computePartialGradient(inputs, output(index), parameters, index)
    val oldScalar = historicalGradients(index)
    (index, scalar, oldScalar)
  }
}


class MiniBatchSagaLS(inputs: CSRMatrix,
                      output: Array[Double],
                      stepSize: Double,
                      iterations_factor: Int,
                      historyRatio: Int,
                      cores: Int,
                      lambda: Double)
  extends MiniBatchSaga(inputs, output, stepSize, iterations_factor, historyRatio, cores, lambda) with LeastSquares


class MiniBatchSagaLogistic(inputs: CSRMatrix,
                            output: Array[Double],
                            stepSize: Double,
                            iterations_factor: Int,
                            historyRatio: Int,
                            cores: Int,
                            lambda: Double)
  extends MiniBatchSaga(inputs, output, stepSize, iterations_factor, historyRatio, cores, lambda) with LogisticRegression


object MiniBatchSaga {
  def main(args: Array[String]) {
    if (args.size != 7) {
      println("Usage: MiniBatchSaga <data><model><stepsize><iterations><history><cores><lambda>")
      System.exit(0)
    }
    val (inputs, output) = args(0) match {
      case "rcv1" => loadDataRCV1()
    }
    val (stepSize, iterationsFactor, historyRatio, cores, lambda) = (args(2).toDouble, args(3).toInt, args(4).toInt, args(5).toInt, args(6).toInt)

    val sls = args(1) match {
      case "LS" => new MiniBatchSagaLS(inputs, output, stepSize, iterationsFactor, historyRatio, cores, lambda)
      case "Logistic" => new MiniBatchSagaLogistic(inputs, output, stepSize, iterationsFactor, historyRatio, cores, lambda)
    }

    sls.run(true, false)
  }
}
