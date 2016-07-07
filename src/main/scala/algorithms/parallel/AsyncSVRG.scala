package algorithms.parallel

import com.google.common.util.concurrent.AtomicDoubleArray
import models.{LeastSquares, LogisticRegression}
import utils._

import scala.concurrent.ExecutionContext.Implicits.global
import scala.concurrent.duration._
import scala.concurrent.{Await, Future}
import scala.util.Random._

/**
  * Async, sparse Saga with compare and swap instructions
  * See CasSGD for parameter explanation.
  */
abstract class AsyncSVRG(inputs: CSRMatrix,
                         output: Array[Double],
                         stepSize: Double,
                         iterationsFactor: Int,
                         historyRatio: Int,
                         cores: Int,
                         lambda: Double,
                         miniBatchSize: Int = 1)
  extends CasSGD(inputs, output, stepSize, iterationsFactor, historyRatio, cores, lambda, miniBatchSize) {

  val atomicReferenceParameters = new AtomicDoubleArray(dimension)

  def run(printLosses: Boolean, verbose: Boolean) {
    // initialization
    initializeParameters()

    for (epoch <- 1 to iterationsFactor / 3) {
      val tempArray = Array.fill[Double](dimension)(0d)
      val currentTime = System.currentTimeMillis()

      // prepare reference gradient
      for (d <- 0 until dimension) {
        atomicReferenceParameters.set(d, atomicParameters.get(d))
        tempArray(d) = atomicParameters.get(d)
      }

      val referenceGradient = atomicCFGParallel(inputs, output, atomicReferenceParameters, cores, splits)._1

      // store uninteresting history (nothing moves in the batch part of SVRG)
      for (t <- 3 * (epoch - 1) * n / historyRatio to (3 * (epoch - 1) + 1) * n / historyRatio) {
        parametersHistory(t) = (currentTime - initialTimeStamp, tempArray, t * renormHR)
      }

      atomicIterationCounter.addAndGet(n / itCtrRatio)

      // launch cores gradient computation in parallel
      val futures = (1 to cores).map(
        _ =>
          if (miniBatchSize == 1) Future(atomicSingleRun(atomicReferenceParameters, referenceGradient, epoch))
          else Future(miniBatchSingleRun(miniBatchSize, atomicReferenceParameters, referenceGradient, epoch))
      )
      Await.ready(Future.sequence(futures), Duration.Inf)
    }

    // compute the losses and print them
    writeLosses(printLosses, verbose)
  }

  // What every thread will run
  def atomicSingleRun(atomicReferenceParameters: AtomicDoubleArray, atomicReferenceGradient: AtomicDoubleArray, nrun: Int) {
    // initialize reusable data structures
    var gc = 0

    for (i <- 1 to ((2 * n) / cores) + 1) {
      // pick a random number
      val index = nextInt(n)

      // compute the partial gradient using the parameters
      val scalar = atomicCPG(inputs, output(index), atomicParameters, index)
      val oldScalar = atomicCPG(inputs, output(index), atomicReferenceParameters, index)

      // take a step
      for (j <- inputs.indPtr(index) until inputs.indPtr(index + 1)) {
        val k = inputs.indices(j)
        val v = inputs.data(j)
        atomicParameters.addAndGet(k,
          - stepSize * (v * (scalar - oldScalar) + inversePi(k) * (atomicReferenceGradient.get(k) + lambda * atomicParameters.get(k))))
      }

      if (i % itCtrRatio == 0) {
        gc = atomicIterationCounter.addAndGet(1)
        val tempArray = Array.fill[Double](dimension)(0d)

        if (gc % renormHR == 0) {
          for (d <- 0 until dimension) {
            tempArray(d) = atomicParameters.get(d)
          }

          parametersHistory(gc /renormHR) = (System.currentTimeMillis() - initialTimeStamp, tempArray, gc)
        }
      }
    }
  }

  // careful, when using mini-batch you can usually rescale the step size by roughly the size of the minibatch
  def miniBatchSingleRun(miniBatchSize: Int, atomicReferenceParameters: AtomicDoubleArray, atomicReferenceGradient: AtomicDoubleArray, nrun: Int) {
    // initialize reusable data structures
    var gc = 0
    val pgdBuffer = Array.fill[Double](miniBatchSize)(0d)
    val inputBuffer = Array.fill[Int](miniBatchSize)(0)

    for (i <- 1 to (2 * n + 1) / (cores * miniBatchSize)) {
      for (j <- 0 until miniBatchSize) {
        // pick a random number
        val index = nextInt(n)

        // compute the partial gradient using the parameters
        val scalar = atomicCPG(inputs, output(index), atomicParameters, index)
        val oldScalar = atomicCPG(inputs, output(index), atomicReferenceParameters, index)

        // store the gradient difference
        pgdBuffer(j) = scalar - oldScalar
        inputBuffer(j) = index
      }

      for (j <- 0 until miniBatchSize;
           index = inputBuffer(j);
           l <- inputs.indPtr(index) until inputs.indPtr(index + 1)) {
        val k = inputs.indices(l)
        val v = inputs.data(l)
        atomicParameters.addAndGet(k,
          -stepSize / miniBatchSize * (v * pgdBuffer(j) + inversePi(k) * (atomicReferenceGradient.get(k) + lambda * atomicParameters.get(k))))
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


class AsyncSVRGLS(inputs: CSRMatrix,
                  output: Array[Double],
                  stepSize: Double,
                  iterations_factor: Int,
                  historyRatio: Int,
                  cores: Int,
                  lambda: Double,
                  miniBatchSize: Int = 1)
  extends AsyncSVRG(inputs, output, stepSize, iterations_factor, historyRatio, cores, lambda, miniBatchSize) with LeastSquares


class AsyncSVRGLogistic(inputs: CSRMatrix,
                        output: Array[Double],
                        stepSize: Double,
                        iterations_factor: Int,
                        historyRatio: Int,
                        cores: Int,
                        lambda: Double,
                        miniBatchSize: Int = 1)
  extends AsyncSVRG(inputs, output, stepSize, iterations_factor, historyRatio, cores, lambda, miniBatchSize) with LogisticRegression


object AsyncSVRG {
  def main(args: Array[String]) {
    if (args.size != 8) {
      println("Usage: AsyncSVRG <data><model><stepsize><iterations><history><cores><lambda><minibatchsize>")
      System.exit(0)
    }
    val (inputs, output) = args(0) match {
      case "rcv1" => loadDataRCV1()
    }
    val (stepSize, iterationsFactor, historyRatio, cores, lambda, miniBatchSize) = (args(2).toDouble, args(3).toInt, args(4).toInt, args(5).toInt, args(6).toDouble, args(7).toInt)

    val sls = args(1) match {
      case "LS" => new AsyncSVRGLS(inputs, output, stepSize, iterationsFactor, historyRatio, cores, lambda, miniBatchSize)
      case "Logistic" => new AsyncSVRGLogistic(inputs, output, stepSize, iterationsFactor, historyRatio, cores, lambda, miniBatchSize)
    }

    sls.run(true, false)
  }
}
