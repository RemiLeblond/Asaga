/**
  * Created by rleblond on 2/23/16.
  */

import algorithms.parallel.AsyncSVRGLogistic
import com.google.common.util.concurrent.AtomicDoubleArray
import utils._
import org.scalatest._

class TestUtils extends FlatSpec with Matchers {
  val (inputs, output) = loadDataRCV1()

  val splits = splitCSR(inputs, 4)
  println(splits.mkString(";"))

  val svrgLogistic = new AsyncSVRGLogistic(inputs, output, 1, 1, 1000000, 10, 1)
  val parameters = Array.fill[Double](inputs.nCols)(1d)
  val gradient1 = svrgLogistic.computeFullGradient(inputs, output, parameters)._1
  val gradient2 = svrgLogistic.computeFullGradientParallel(inputs, output, parameters, 4, splits)

  "both gradients" should "be equal" in {
    for (d <- 0 until parameters.length) {
      // small difference due to double precision error
      if (gradient1(d) != gradient2(d)) {
        // println(gradient1(d), gradient2(d))
      }
      gradient1(d) - gradient2(d) should be < 1e-15
    }
  }

  val atomicParameters = new AtomicDoubleArray(inputs.nCols)
  for (d <- 0 until inputs.nCols) {
    atomicParameters.set(d, 0d)
  }

  val (atomicGradient1, historicalGradient1) = svrgLogistic.atomicCFG(inputs, output, atomicParameters)
  val (atomicGradient2, historicalGradient2) = svrgLogistic.atomicCFGParallel(inputs, output, atomicParameters, 4, splits)

  "both atomic gradients" should "be equal" in {
    for (d <- 0 until parameters.length) {
      // small difference due to double precision error
      atomicGradient1.get(d) - atomicGradient2.get(d) should be < 1e-15
    }
  }

  "both historical gradients" should "be equal" in {
    for (i <- 0 until inputs.nRows) {
      // small difference due to double precision error
      historicalGradient1.get(i) - historicalGradient2.get(i) should be < 1e-15
    }
  }
}
