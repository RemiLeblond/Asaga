package algorithms.parallel

import com.google.common.util.concurrent.AtomicDoubleArray
import models.{LeastSquares, LogisticRegression}
import utils._

import scala.collection.mutable
import scala.concurrent.ExecutionContext.Implicits.global
import scala.concurrent.duration._
import scala.concurrent.{Await, Future}
import scala.util.Random._

/**
  * Created by rleblond on 1/15/16.
  * Async, sparse Saga with compare and swap instructions
  * See CasSGD for parameter explanation.
  */
abstract class AsyncSaga(inputs: CSRMatrix,
                         output: Array[Double],
                         stepSize: Double,
                         iterationsFactor: Int,
                         historyRatio: Int,
                         cores: Int,
                         lambda: Double,
                         miniBatchSize: Int = 1)
  extends CasSGD(inputs, output, stepSize, iterationsFactor, historyRatio, cores, lambda, miniBatchSize) {

  def run(printLosses: Boolean, verbose: Boolean) {
    // initialization
    initializeParameters()

    // First pass with constant step-size SGD in order to compute the historical gradient at no initial cost
    val atomicCurrentAverageGradient = new AtomicDoubleArray(inputs.nCols)
    val atomicHistoricalGradients = new AtomicDoubleArray(inputs.nRows)

    val initFutures = (0 until cores).map(
      i => Future(atomicFirstRun(atomicCurrentAverageGradient, atomicHistoricalGradients, splits, i))
    )
    Await.ready(Future.sequence(initFutures), Duration.Inf)

    //launch cores gradient computation in parallel
    val futures = (1 to cores).map(
      _ =>
        if (miniBatchSize == 1) Future(atomicSingleRun(atomicCurrentAverageGradient, atomicHistoricalGradients))
        else Future(miniBatchSingleRun(miniBatchSize, atomicCurrentAverageGradient, atomicHistoricalGradients))
    )
    Await.ready(Future.sequence(futures), Duration.Inf)

    // compute the losses and print them
    writeLosses(printLosses, verbose)
  }

  // What every thread will run except for the first pass
  def atomicSingleRun(atomicCAG: AtomicDoubleArray, atomicHG: AtomicDoubleArray) {
    // Initialize global counter
    var gc = n

    for (i <- 1 to (iterations + 1) / cores) {
      // pick a random number
      val index = nextInt(n)

      // compute the partial gradient using the parameters
      val scalar = atomicCPG(inputs, output(index), atomicParameters, index)
      val oldScalar = atomicHG.get(index)

      // take a step
      for (j <- inputs.indPtr(index) until inputs.indPtr(index + 1)) {
        val k = inputs.indices(j)
        val v = inputs.data(j)
        atomicParameters.addAndGet(k,
          - stepSize * (v * (scalar - oldScalar) + inversePi(k) * (atomicCAG.get(k) + lambda * atomicParameters.get(k))))
      }

      // update the average gradient and the historical one
      atomicHG.addAndGet(index, scalar - oldScalar)
      for (j <- inputs.indPtr(index) until inputs.indPtr(index + 1)) {
        val k = inputs.indices(j)
        val v = inputs.data(j)
        atomicCAG.addAndGet(k, (scalar - oldScalar) * v / n)
      }

      // store history for plotting
      if (i % itCtrRatio == 0) {
        gc = atomicIterationCounter.addAndGet(1)
        val tempArray = Array.fill[Double](dimension)(0d)

        if (gc % renormHR == 0) {
          for (d <- 0 until dimension) {
            tempArray(d) = atomicParameters.get(d)
          }
          parametersHistory(gc / renormHR) = (System.currentTimeMillis() - initialTimeStamp, tempArray, gc)
        }
      }
    }
  }

  // first pass
  def atomicFirstRun(atomicCAG: AtomicDoubleArray, atomicHG: AtomicDoubleArray, splits: Array[Int], split: Int) = {
    var gc = 0

    for (index <- splits(split) until splits(split + 1)) {
      // compute the partial gradient using the parameters
      val scalar = atomicCPG(inputs, output(index), atomicParameters, index)

      // take a step
      for (j <- inputs.indPtr(index) until inputs.indPtr(index + 1)) {
        val k = inputs.indices(j)
        val v = inputs.data(j)
        atomicParameters.addAndGet(k, - stepSize * (v * scalar + lambda * inversePi(k) * atomicParameters.get(k)))
        atomicCAG.addAndGet(k, scalar * v / n)
      }

      atomicHG.addAndGet(index, scalar)

      if (index % itCtrRatio == 0) {
        gc = atomicIterationCounter.addAndGet(1)
        val tempArray = Array.fill[Double](dimension)(0d)

        if (gc % renormHR == 0) {
          for (d <- 0 until dimension) {
            tempArray(d) = atomicParameters.get(d)
          }
          parametersHistory(gc / renormHR) = (System.currentTimeMillis() - initialTimeStamp, tempArray, gc)
        }
      }
    }
  }

  // careful, when using mini-batch you can usually rescale the step size by roughly the size of the minibatch
  def miniBatchSingleRun(miniBatchSize: Int, atomicCAG: AtomicDoubleArray, atomicHG: AtomicDoubleArray) {
    // Initialize reusable data structures
    var gc = 0
    val pgdBuffer = Array.fill[Double](miniBatchSize)(0d)
    val inputBuffer = Array.fill[Int](miniBatchSize)(0)
    val updates = Array.fill[Double](dimension)(0d)
    val cagUpdates = Array.fill[Double](dimension)(0d)
    var dimensions = List.empty[Int] // used to only look at the subset of relevant dimensions

    for (i <- 1 to (iterations + 1) / (cores * miniBatchSize)) {
      for (j <- 0 until miniBatchSize) {
        // pick a random number
        val index = nextInt(n)

        // compute the partial gradient using the parameters
        val scalar = atomicCPG(inputs, output(index), atomicParameters, index)
        val oldScalar = atomicHG.get(index)

        // store the partial gradients
        pgdBuffer(j) = scalar - oldScalar
        inputBuffer(j) = index
      }

      // take a step
      for (j <- 0 until miniBatchSize) {
        atomicHG.addAndGet(inputBuffer(j), pgdBuffer(j))
        val index = inputBuffer(j)
        for (l <- inputs.indPtr(index) until inputs.indPtr(index + 1)) {
          val k = inputs.indices(l)
          val v = inputs.data(l)
          dimensions = k::dimensions

          updates(k) -= stepSize / miniBatchSize * (v * pgdBuffer(j) + inversePi(k) * (atomicCAG.get(k) + lambda * atomicParameters.get(k)))
          cagUpdates(k) += v * pgdBuffer(j) / n
        }

        for (k <- dimensions) {
          atomicParameters.addAndGet(k, updates(k))
          atomicCAG.addAndGet(k, cagUpdates(k))
          updates(k) = 0d
          cagUpdates(k) = 0d
        }
        dimensions = List.empty[Int]
      }

      if (i * miniBatchSize % itCtrRatio == 0 || i * miniBatchSize % itCtrRatio > ((i + 1) * miniBatchSize - 1) % itCtrRatio) { // means we passed itCtrRatio in this minibatch
        gc = atomicIterationCounter.addAndGet(1)
        val tempArray = Array.fill[Double](dimension)(0d)

        if (gc % renormHR == 0) {
          for (d <- 0 until dimension) {
            tempArray(d) = atomicParameters.get(d)
          }
          parametersHistory(gc / renormHR) = (System.currentTimeMillis() - initialTimeStamp, tempArray, gc)
        }
      }
    }
  }
}


class AsyncSagaLS(inputs: CSRMatrix,
                  output: Array[Double],
                  stepSize: Double,
                  iterations_factor: Int,
                  historyRatio: Int,
                  cores: Int,
                  lambda: Double,
                  miniBatchSize: Int = 1)
  extends AsyncSaga(inputs, output, stepSize, iterations_factor, historyRatio, cores, lambda, miniBatchSize) with LeastSquares


class AsyncSagaLogistic(inputs: CSRMatrix,
                        output: Array[Double],
                        stepSize: Double,
                        iterations_factor: Int,
                        historyRatio: Int,
                        cores: Int,
                        lambda: Double,
                        miniBatchSize: Int = 1)
  extends AsyncSaga(inputs, output, stepSize, iterations_factor, historyRatio, cores, lambda, miniBatchSize) with LogisticRegression


/**
  * Entry point to launch the method with the compiled JAR.
  */
object AsyncSaga {
  def main(args: Array[String]) {
    if (args.size != 8) {
      println("Usage: AsyncSaga <data><model><stepsize><iterations><history><cores><lambda><minibatchsize>")
      System.exit(0)
    }
    val (inputs, output) = args(0) match {
      case "rcv1" => loadDataRCV1()
      case "rcv1full" => loadDataRCV1Full()
      case "urls" => loadDataURL()
      case "covtype" => loadDataCovtype()
    }
    val (stepSize, iterationsFactor, historyRatio, cores, lambda, miniBatchSize) = (args(2).toDouble, args(3).toInt, args(4).toInt, args(5).toInt, args(6).toDouble, args(7).toInt)

    val sls = args(1) match {
      case "LS" => new AsyncSagaLS(inputs, output, stepSize, iterationsFactor, historyRatio, cores, lambda, miniBatchSize)
      case "Logistic" => new AsyncSagaLogistic(inputs, output, stepSize, iterationsFactor, historyRatio, cores, lambda, miniBatchSize)
    }

    sls.run(true, false)
  }
}
