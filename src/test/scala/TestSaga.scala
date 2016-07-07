/**
  * Test the SAGA algorithm.
  */

import algorithms.parallel.AsyncSagaLogistic
import algorithms.sequential.{LaggedSagaLogistic, SagaLogistic, SparseSagaLogistic}
import utils._
import org.scalatest._

class TestSaga extends FlatSpec with Matchers {
  val optimal_loss_RCV1 = 0.20327841000409613
  val (inputs, output) = loadDataRCV1()
  val stepsize = 1.0
  val iterations_factor = 10
  val lambda = 1.0 / inputs.nRows

  "Saga" should "reach optimal value" in {
    // in this case we use lower iterations and
    // lower tolerance just because otherwise it takes forever
    val basicSagaLogistic = new SagaLogistic(inputs, output, stepsize, iterations_factor / 10, 5000, lambda)
    basicSagaLogistic.run(false, false)
    val sagaLoss = basicSagaLogistic.computeFullLoss(inputs, output, basicSagaLogistic.parameters, lambda)
    val diff = sagaLoss - optimal_loss_RCV1
    println("Difference", diff)
    diff should be < 1.0
  }

  "SparseSaga" should "reach optimal value" in {
    val sparseSagaLogistic = new SparseSagaLogistic(inputs, output, stepsize, iterations_factor, 5000, lambda)
    sparseSagaLogistic.run(false, false)
    val sparseSagaLoss = sparseSagaLogistic.computeFullLoss(inputs, output, sparseSagaLogistic.parameters, lambda)
    val sparseDiff = sparseSagaLoss - optimal_loss_RCV1
    println("Difference", sparseDiff)
    sparseDiff should be < 1e-3d
  }

  "LaggedSaga" should "reach optimal value" in {
    val laggedSagaLogistic = new LaggedSagaLogistic(inputs, output, stepsize, iterations_factor, 5000, lambda)
    laggedSagaLogistic.run(false, false)
    val sparseSagaLoss = laggedSagaLogistic.computeFullLoss(inputs, output, laggedSagaLogistic.parameters, lambda)
    val laggedDiff = sparseSagaLoss - optimal_loss_RCV1
    println("Difference", laggedDiff)
    laggedDiff should be < 1e-3d
  }

  "AsyncSaga" should "reach optimal value" in {
    val nCores = 3
    val asyncSagaLogistic = new AsyncSagaLogistic(inputs, output, stepsize, iterations_factor, 5000, nCores, lambda)
    asyncSagaLogistic.run(false, false)
    val sparseSagaLoss = asyncSagaLogistic.atomicCFL(inputs, output, asyncSagaLogistic.atomicParameters, lambda)
    val asyncDiff = sparseSagaLoss - optimal_loss_RCV1
    println("Difference", asyncDiff)
    asyncDiff should be < 1e-3d
  }

  "AsyncSaga minibatch" should "reach optimal value" in {
    val nCores = 3
    val asyncSagaLogistic = new AsyncSagaLogistic(inputs, output, stepsize, iterations_factor * 10, 5000, nCores, lambda,
      miniBatchSize=10)
    asyncSagaLogistic.run(false, false)
    val sparseSagaLoss = asyncSagaLogistic.atomicCFL(inputs, output, asyncSagaLogistic.atomicParameters, lambda)
    val asyncDiff = sparseSagaLoss - optimal_loss_RCV1
    println("Difference", asyncDiff)
    asyncDiff should be < 1e-3d
  }
}
