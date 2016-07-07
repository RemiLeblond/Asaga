package algorithms.parallel

import models.{LogisticRegression, LeastSquares}
import utils._

import scala.collection.mutable
import scala.concurrent.ExecutionContext.Implicits.global
import scala.concurrent.duration.Duration
import scala.concurrent.{Await, Future}
import scala.util.Random._

/**
  * Created by rleblond on 2/11/16.
  * Async SGD
  * For parameter details, see CasSGD
  */
abstract class Hogwild(inputs: CSRMatrix,
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

    //launch cores gradient computation in parallel
    val futures = (1 to cores).map(
      _ =>
        if (miniBatchSize == 1) Future(atomicSingleRun())
        else Future(miniBatchSingleRun(miniBatchSize))
    )
    Await.ready(Future.sequence(futures), Duration.Inf)

    // compute the losses and print them
    writeLosses(printLosses, verbose)
  }

  // What every thread will run
  def atomicSingleRun() {
    // Initialize reusable data structures
    var gc = 0

    for (i <- 1 to (iterations + 1) / cores) {
      // pick a random number
      val index = nextInt(n)

      // compute the partial gradient using the parameters
      val scalar = atomicCPG(inputs, output(index), atomicParameters, index)

      // take a step
      for (j <- inputs.indPtr(index) until inputs.indPtr(index + 1)) {
        val k = inputs.indices(j)
        val v = inputs.data(j)
        atomicParameters.addAndGet(k, - stepSize * (v * scalar + lambda * inversePi(k) * atomicParameters.get(k)))
      }

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

  // careful, when using mini-batch you can usually rescale the step size by roughly the size of the minibatch
  def miniBatchSingleRun(miniBatchSize: Int) {
    // Initialize reusable data structures
    var gc = 0
    val pgBuffer = Array.fill[Double](miniBatchSize)(0d)
    val inputBuffer = Array.fill[Int](miniBatchSize)(0)
    val updatesMap = mutable.Map.empty[Int, Double]

    for (i <- 1 to (iterations + 1)/ (cores * miniBatchSize)) {
      for (j <- 0 until miniBatchSize) {
        // pick a random number
        val index = nextInt(n)

        // compute the partial gradient using the parameters
        val scalar = atomicCPG(inputs, output(index), atomicParameters, index)

        // store the partial gradients
        pgBuffer(j) = scalar
        inputBuffer(j) = index
      }

      for (j <- 0 until miniBatchSize;
           index = inputBuffer(j);
           l <- inputs.indPtr(index) until inputs.indPtr(index + 1)) {

        val k = inputs.indices(l)
        val v = inputs.data(l)
        val previousV = updatesMap.getOrElse(k, 0d)
        updatesMap(k) = previousV - stepSize / miniBatchSize * (v * pgBuffer(j) + lambda * inversePi(k) * atomicParameters.get(k))
      }

      for ((k,v) <- updatesMap) {
        atomicParameters.addAndGet(k, v)
      }

      updatesMap.clear()

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


class HogwildLS(inputs: CSRMatrix,
                output: Array[Double],
                stepSize: Double,
                iterations_factor: Int,
                historyRatio: Int,
                cores: Int,
                lambda: Double,
                miniBatchSize: Int = 1)
  extends Hogwild(inputs, output, stepSize, iterations_factor, historyRatio, cores, lambda, miniBatchSize) with LeastSquares


class HogwildLogistic(inputs: CSRMatrix,
                      output: Array[Double],
                      stepSize: Double,
                      iterations_factor: Int,
                      historyRatio: Int,
                      cores: Int,
                      lambda: Double,
                      miniBatchSize: Int = 1)
  extends Hogwild(inputs, output, stepSize, iterations_factor, historyRatio, cores, lambda, miniBatchSize) with LogisticRegression


object Hogwild {
  def main(args: Array[String]) {
    if (args.size != 8) {
      println("Usage: Hogwild <data><model><stepsize><iterations><history><cores><lambda><minibatchsize>")
      System.exit(0)
    }
    val (inputs, output) = args(0) match {
      case "rcv1" => loadDataRCV1()
    }
    val (stepSize, iterationsFactor, historyRatio, cores, lambda, miniBatchSize) = (args(2).toDouble, args(3).toInt, args(4).toInt, args(5).toInt, args(6).toDouble, args(7).toInt)

    val sls = args(1) match {
      case "LS" => new HogwildLS(inputs, output, stepSize, iterationsFactor, historyRatio, cores, lambda, miniBatchSize)
      case "Logistic" => new HogwildLogistic(inputs, output, stepSize, iterationsFactor, historyRatio, cores, lambda, miniBatchSize)
    }

    sls.run(true, false)
  }
}
