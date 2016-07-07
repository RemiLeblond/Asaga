package models

import com.google.common.util.concurrent.AtomicDoubleArray
import utils._
/**
  * Created by rleblond on 1/14/16.
  * Least-squares computation to be mixed in the algorithm of choice.
  */

trait LeastSquares extends Model {
  def computePartialLoss(inputs: CSRMatrix, output: Double, parameters: Array[Double], nRow: Int): Double = {
    Math.pow(sqrtPartialLoss(inputs, output, parameters, nRow), 2)
  }

  def computePartialGradient(inputs: CSRMatrix, output: Double, parameters: Array[Double], nRow: Int): Double = {
    2 * sqrtPartialLoss(inputs, output, parameters, nRow)
  }

  def sqrtPartialLoss(inputs: CSRMatrix, output: Double, parameters: Array[Double], nRow: Int): Double = {
    val ip = innerProduct(inputs, parameters, nRow)
    ip - output
  }

  /**
    * Atomic versions of the above.
    */

  def atomicCPL(inputs: CSRMatrix, output: Double, atomicParameters: AtomicDoubleArray, nRow: Int): Double = {
    Math.pow(atomicSPL(inputs, output, atomicParameters, nRow), 2)
  }

  def atomicCPG(inputs: CSRMatrix, output: Double, atomicParameters: AtomicDoubleArray, nRow: Int): Double = {
    2 * atomicSPL(inputs, output, atomicParameters, nRow)
  }

  def atomicSPL(inputs: CSRMatrix, output: Double, atomicParameters: AtomicDoubleArray, nRow: Int): Double = {
    val ip = atomicInnerProduct(inputs, atomicParameters, nRow)
    ip - output
  }
}
