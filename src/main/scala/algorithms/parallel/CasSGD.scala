package algorithms.parallel

import java.io.{File, PrintWriter}
import java.util.concurrent.atomic.AtomicInteger

import com.google.common.util.concurrent.AtomicDoubleArray
import utils._

import scala.concurrent.duration.Duration
import scala.concurrent.{Await, Future}
import scala.concurrent.ExecutionContext.Implicits.global

/**
  * Created by rleblond on 2/11/16.
  */

/**
  * Base-class for every asynchronous SGD-type algorithm using compare-and-swap semantics.
  * The specific loss is mixed in via a trait in package models.
  * @param inputs Input represented as a CSRMatrix, usually built from the utils#loadDataCSR method.
  * @param output Output represented as an Array of Double.
  * @param stepSize Learning rate of the algorithm
  * @param iterationsFactor Number of epochs
  * @param historyRatio Fraction of iterations used for plotting
  * @param cores Number of threads allocated for this computation
  * @param lambda Regularization parameter
  * @param miniBatchSize Size of the mini-batch. Default = 1, i.e. full asynchronous.
  */
abstract class CasSGD(inputs: CSRMatrix,
                      output: Array[Double],
                      stepSize: Double,
                      iterationsFactor: Int,
                      historyRatio: Int,
                      cores: Int,
                      lambda: Double,
                      miniBatchSize: Int = 1) {

  val initialParameterValue = 0d
  val dimension = inputs.nCols
  val atomicParameters = new AtomicDoubleArray(dimension)
  val n = inputs.nRows
  val iterations = n * iterationsFactor
  var initialTimeStamp = 0L

  // global iteration counter. Only used for plotting.
  val atomicIterationCounter = new AtomicInteger(0)
  // We don't want to update the global counter at every iteration (thus making it the most conflicted structure).
  val itCtrRatio = 100
  require(miniBatchSize <= itCtrRatio, "mini-batch should be smaller than history granularity")
  val renormHR = historyRatio / itCtrRatio  // Renormalize history ratio so behaviour is as expected

  // Used to store the parameters and compute the actual suboptimality in the end
  val parametersHistory = Array.fill[(Double, Array[Double], Int)](iterations / historyRatio + 1)(0L, Array.fill[Double](dimension)(0L), 0)

  // compute the pi probabilities
  val inversePi = computeInversePis(inputs, dimension)

  // for parallelization
  val splits = splitCSR(inputs, cores)

  def initializeParameters() {
    for (i <- 0 until dimension) {
      atomicParameters.set(i, initialParameterValue)
    }
    parametersHistory(0) = (0, Array.fill[Double](dimension)(initialParameterValue), 0)
    atomicIterationCounter.set(0)
    initialTimeStamp = System.currentTimeMillis()
  }

  // full gradient computation
  def atomicCFG(inputs: CSRMatrix, output: Array[Double], atomicParameters: AtomicDoubleArray): (AtomicDoubleArray, AtomicDoubleArray)

  // individual gradient computation
  def atomicCPG(inputs: CSRMatrix, output: Double, atomicParameters: AtomicDoubleArray, nRow: Int): Double

  // full loss computation
  def atomicCFL(inputs: CSRMatrix, output: Array[Double], atomicParameters: AtomicDoubleArray, lambda: Double): Double

  // parallelized full gradient computation
  def atomicCFGParallel(inputs: CSRMatrix, output: Array[Double], atomicParameters: AtomicDoubleArray, cores: Int, splits: Array[Int]): (AtomicDoubleArray, AtomicDoubleArray)

  // full loss computation
  def computeFullLoss(inputs: CSRMatrix, output: Array[Double], parameters: Array[Double], lambda: Double): Double

  // history of losses computation
  def computeAllLosses(historyLosses: Array[String]) {
    val availableCores = Runtime.getRuntime.availableProcessors()
    val futures = (0 until availableCores).map(
      i => Future(computeSplitLosses(historyLosses, i, availableCores))
    )
    Await.ready(Future.sequence(futures), Duration.Inf)
  }

  // for parallel loss computation
  def computeSplitLosses(historyLosses: Array[String], i: Int, cores: Int) {
    for (j <- 0 until historyLosses.size) {
      if (j % cores == i) {
        historyLosses(j) = parametersHistory(j)._1 + "\t" + computeFullLoss(inputs, output, parametersHistory(j)._2, lambda) + "\t" + parametersHistory(j)._3
      }
    }
  }

  def writeLosses(printLosses: Boolean, verbose: Boolean) {
    // sanity check
    val afterUpdatesTS = System.currentTimeMillis()
    val loss = atomicCFL(inputs, output, atomicParameters, lambda)
    val finalTS = System.currentTimeMillis()
    if (verbose) {
      println(s"final loss $loss")
      println(s"Time spent in iterations ${afterUpdatesTS - initialTimeStamp}")
      println(s"Time spent evaluating objective ${finalTS - afterUpdatesTS}")
    }

    if (printLosses) {
      // print out file
      val timestamp = System.currentTimeMillis()
      val historyLosses = Array.fill[String](parametersHistory.size)("")
      computeAllLosses(historyLosses)
      val name = timestamp + "_" + this.getClass.getSimpleName + ".csv"
      val fileName = sys.env("SAGA_DATA_DIR") + "/" + name
      val writer = new PrintWriter(new File(fileName))
      writer.write(historyLosses.mkString("\n"))
      writer.close()
    }
  }

  def run(printLosses: Boolean, verbose: Boolean)
}
