package algorithms

import java.io.{File, PrintWriter}

import scala.concurrent.duration.Duration
import scala.concurrent.{Await, Future}
import scala.concurrent.ExecutionContext.Implicits.global
import utils._

/**
 * Created by rleblond on 1/27/16.
 */

/**
  * Base class for all SGD-like methods including Saga and SVRG
  * @param inputs Input represented as a CSRMatrix, usually built from the utils#loadDataCSR method.
  * @param output Output represented as an Array of Double.
  * @param stepSize Learning rate of the algorithm
  * @param iterationsFactor Number of epochs
  * @param historyRatio Fraction of iterations used for plotting
  * @param lambda Regularization parameter
  */
abstract class StochasticGradientDescent(inputs: CSRMatrix,
                                         output: Array[Double],
                                         stepSize: Double,
                                         iterationsFactor: Int,
                                         historyRatio: Int,
                                         lambda: Double) {

  val initialParameterValue = 0d
  val dimension = inputs.nCols
  val parameters = Array.fill[Double](dimension)(initialParameterValue)
  val n = inputs.nRows
  val iterations = n * iterationsFactor
  var initialTimeStamp = 0L

  // Used to store the parameters and compute the actual suboptimality in the end
  val parametersHistory = Array.fill[(Double, Array[Double], Int)](iterations / historyRatio + 1)(0L, Array.fill[Double](dimension)(0L), 0)

  // batch gradient computation
  def computeFullGradient(inputs: CSRMatrix, output: Array[Double], parameters: Array[Double]): (Array[Double], Array[Double])

  // individual gradient computation
  def computePartialGradient(inputs: CSRMatrix, output: Double, parameters: Array[Double], nRow: Int): Double

  // full loss computation
  def computeFullLoss(inputs: CSRMatrix, output: Array[Double], parameters: Array[Double], lambda: Double): Double

  // computation of all losses of the algorithm for plotting
  def computeAllLosses(historyLosses: Array[String]) {
    val availableCores = 2  // Runtime.getRuntime.availableProcessors()
    val futures = (0 until availableCores).map(
      i => Future(computeSplitLosses(historyLosses, i, availableCores))
    )
    Await.ready(Future.sequence(futures), Duration.Inf)
  }

  // helper method for parallel computation of losses
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
    val loss = computeFullLoss(inputs, output, parameters, lambda)
    val finalTS = System.currentTimeMillis()
    if (verbose) {
      println(s"final loss $loss")
      println(s"Time spent in iterations ${afterUpdatesTS - initialTimeStamp}")
      println(s"Time spent evaluating objective ${finalTS - afterUpdatesTS}")
    }

    if (printLosses) {
      // Print out file
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