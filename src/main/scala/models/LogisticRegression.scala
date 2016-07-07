package models

import com.google.common.util.concurrent.AtomicDoubleArray
import utils._

/**
  * Created by rleblond on 1/14/16.
  * Least-squares computation to be mixed in the algorithm of choice.
  */
trait LogisticRegression extends Model {
  def computePartialLoss(inputs: CSRMatrix, output: Double, parameters: Array[Double], nRow: Int): Double = {
    val ip = innerProduct(inputs, parameters, nRow)
    val t = output * ip
    if (t > 0) {Math.log(1 + Math.exp(-t))}
    else {Math.log(Math.exp(t) + 1) - t}
  }

  def computePartialGradient(inputs: CSRMatrix, output: Double, parameters: Array[Double], nRow: Int): Double = {
    val ip = innerProduct(inputs, parameters, nRow)
    val t = output * ip
    if (t < 0) {- output / (1 + Math.exp(t))}
    else {- output * Math.exp(-t) / (1 + Math.exp(-t))}
  }

  /**
    * atomic versions of the above
    */

  def atomicCPL(inputs: CSRMatrix, output: Double, atomicParameters: AtomicDoubleArray, nRow: Int): Double = {
    val ip = atomicInnerProduct(inputs, atomicParameters, nRow)
    val t = output * ip

    if (t > 0) {Math.log(1 + Math.exp(-t))}
    else {Math.log(Math.exp(t) + 1) - t}
  }

  def atomicCPG(inputs: CSRMatrix, output: Double, atomicParameters: AtomicDoubleArray, nRow: Int): Double = {
    val ip = atomicInnerProduct(inputs, atomicParameters, nRow)
    val t = output * ip

    if (t < 0) {- output / (1 + Math.exp(t))}
    else {- output * Math.exp(-t) / (1 + Math.exp(-t))}
  }
}
