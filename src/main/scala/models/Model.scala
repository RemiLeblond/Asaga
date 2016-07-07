package models

import com.google.common.util.concurrent.AtomicDoubleArray
import utils.CSRMatrix

import scala.concurrent.duration.Duration
import scala.concurrent.{Await, Future}
import scala.concurrent.ExecutionContext.Implicits.global

/**
  * Created by rleblond on 2/9/16.
  * Base trait for models.
  */
trait Model {
  /**
    * Non thread-safe methods
    */

  def computePartialLoss(inputs: CSRMatrix, output: Double, parameters: Array[Double], nRow: Int): Double

  def computeFullLoss(inputs: CSRMatrix, output: Array[Double], parameters: Array[Double], lambda: Double = 0d): Double = {
    val loss = (for (i <- 0 until inputs.nRows) yield computePartialLoss(inputs, output(i), parameters, i)).sum
    loss / inputs.nRows + 0.5 * lambda * parameters.map(d => d * d).sum
  }

  def computePartialGradient(inputs: CSRMatrix, output: Double, parameters: Array[Double], nRow: Int): Double

  def computeFullGradient(inputs: CSRMatrix, output: Array[Double], parameters: Array[Double], firstIndex: Int, lastIndex: Int): (Array[Double], Array[Double]) = {
    val dimension = inputs.nCols
    val n = inputs.nRows
    val fullGradient = Array.fill[Double](dimension)(0d)
    val historicalGradients = Array.fill[Double](n)(0d)

    // sparse updates
    for (i <- firstIndex until lastIndex) {
      val grad = computePartialGradient(inputs, output(i), parameters, i)

      for (j <- inputs.indPtr(i) until inputs.indPtr(i + 1)) {
        val k = inputs.indices(j)
        val v = inputs.data(j)
        fullGradient(k) += grad * v
      }

      historicalGradients(i) = grad
    }
    (fullGradient.map(_ / n), historicalGradients)
  }

  def computeFullGradient(inputs: CSRMatrix, output: Array[Double], parameters: Array[Double]): (Array[Double], Array[Double]) =
    computeFullGradient(inputs, output, parameters, 0, inputs.nRows)

  def computeFullGradientParallel(inputs: CSRMatrix, output: Array[Double], parameters: Array[Double], cores: Int, splits: Array[Int]): Array[Double] = {
    val futures = (0 until cores).map(
      i => Future(computeFullGradient(inputs, output, parameters, splits(i), splits(i + 1))._1)
    )
    val splitGradients = Await.result(Future.sequence(futures), Duration.Inf)
    val gradient = Array.fill[Double](inputs.nCols)(0d)
    for (splitGr <- splitGradients; d <- 0 until inputs.nCols) {
      gradient(d) += splitGr(d)
    }

    gradient
  }

  /**
    * Atomic methods
    */

  def atomicCPL(inputs: CSRMatrix, output: Double, atomicParameters: AtomicDoubleArray, nRow: Int): Double

  def atomicCPG(inputs: CSRMatrix, output: Double, atomicParameters: AtomicDoubleArray, nRow: Int): Double

  def atomicCFL(inputs: CSRMatrix, output: Array[Double], atomicParameters: AtomicDoubleArray, lambda: Double): Double = {
    val loss = (for (i <- 0 until inputs.nRows) yield atomicCPL(inputs, output(i), atomicParameters, i)).sum
    val regularization = (for (i <- 0 until inputs.nCols) yield Math.pow(atomicParameters.get(i), 2)).sum
    loss / inputs.nRows + 0.5 * lambda * regularization
  }

  def atomicCFG(inputs: CSRMatrix, output: Array[Double], atomicParameters: AtomicDoubleArray, firstIndex: Int, lastIndex: Int): (AtomicDoubleArray, AtomicDoubleArray) = {
    val dimension = inputs.nCols
    val n = inputs.nRows
    val fullGradient = new AtomicDoubleArray(dimension)
    val historicalGradients = new AtomicDoubleArray(n)

    // sparse updates
    for (i <- firstIndex until lastIndex) {
      val grad = atomicCPG(inputs, output(i), atomicParameters, i)

      // update my global gradient
      for (j <- inputs.indPtr(i) until inputs.indPtr(i + 1)) {
        val k = inputs.indices(j)
        val v = inputs.data(j)
        fullGradient.addAndGet(k, grad * v / n)
      }
      // fill the historical gradients thing
      historicalGradients.set(i, grad)
    }
    (fullGradient, historicalGradients)
  }

  def atomicCFG(inputs: CSRMatrix, output: Array[Double], atomicParameters: AtomicDoubleArray): (AtomicDoubleArray, AtomicDoubleArray) =
    atomicCFG(inputs, output, atomicParameters, 0, inputs.nRows)


  def atomicCFGParallel(inputs: CSRMatrix, output: Array[Double], atomicParameters: AtomicDoubleArray, cores: Int, splits: Array[Int]): (AtomicDoubleArray, AtomicDoubleArray) = {
    val gradient = new AtomicDoubleArray(inputs.nCols)
    val historicalGradients = new AtomicDoubleArray(inputs.nRows)
    // initialize to zero
    for(d <- 0 until inputs.nCols){
      gradient.set(d, 0.0)
    }
    val futures = (0 until cores).map(
      i => Future(atomicCFG(inputs, output, atomicParameters, splits(i), splits(i + 1)))
    )
    val splitGradients = Await.result(Future.sequence(futures), Duration.Inf)
    for ((splitG, splitHg) <- splitGradients) {
      for (d <- 0 until inputs.nCols) {
        gradient.addAndGet(d, splitG.get(d))
      }

      for (i <- 0 until inputs.nRows) {
        historicalGradients.addAndGet(i, splitHg.get(i))
      }
    }
    (gradient, historicalGradients)
  }
}
