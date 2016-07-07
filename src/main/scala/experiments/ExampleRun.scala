package experiments

import algorithms.parallel._
import algorithms.sequential._
import utils._

object ExampleRun {
  def main(args: Array[String]) {
    val (inputs, output) = loadDataRCV1()
    val asls = new AsyncSagaLogistic(
      inputs = inputs,
      output = output,
      stepSize = 0.1,
      iterations_factor = 50,
      historyRatio = 300000,
      cores = 10,
      lambda = 1e-3,
      miniBatchSize = 10)

    // warm-up run to account for JVM optimization at runtime
    asls.run(false, true)

    System.gc()
    asls.run(true, true)
  }
}
