package utils

import java.io.{File, PrintWriter}

import scala.util.Random

/**
  * Created by rleblond on 5/3/16.
  */
object ControlledDataset {

  def main(args: Array[String]) {
    val dataset = generateDataset(1000, 500)
    printDatasetToFile(dataset._1, dataset._2, dataset._3, "test")
  }

  // generates a dense dataset
  def generateDataset(n: Int, d: Int): (Array[Map[Int, Double]], Array[Double], Array[Double]) = {
    val parameters = Array.fill[Double](d)(Random.nextDouble() * 2.0 - 1.0)
    val inputs = (1 to n).map(x => randomInput(d)).toArray
    val outputs = inputs.map(randomOutput(_, parameters))
    val normalizedInputs = normalizeInput(inputs)
    (normalizedInputs, outputs, parameters)
  }

  def randomInput(d: Int): Map[Int, Double] = (1 to d).zip(Array.fill[Double](d)(Random.nextDouble())).toMap

  def randomOutput(input: Map[Int, Double], parameters: Array[Double]): Double = {
    val prod = input.map({case (x,y) => parameters(x - 1) * y}).sum + Random.nextGaussian()
    if (prod > 0) 1.0 else -1.0
  }

  def normalizeInput(input: Array[Map[Int, Double]]): Array[Map[Int, Double]] = {
    val maxSquaredNorm = input.map(_.map({case (x,y) => y * y}).sum).max
    input.map(_.map({case (x,y) => x -> y / Math.sqrt(maxSquaredNorm)}))
  }

  def printDataset(inputs: Array[Map[Int, Double]], outputs: Array[Double]) = {
    require(inputs.size == outputs.size)

    for (i <- 0 until inputs.size) {
      print(outputs(i))
      print(" ")
      println(inputs(i).map({case (x,y) => x + ":" + y}).mkString(" "))
    }
  }

  def printDatasetToFile(inputs: Array[Map[Int, Double]], outputs: Array[Double], parameters: Array[Double], filename: String): Unit = {
    require(inputs.size == outputs.size)
    val writer = new PrintWriter(new File(filename + "data"))
    val writerp = new PrintWriter(new File(filename + "parameters"))
    val stringbuffer = new StringBuilder()

    for (i <- 0 until inputs.size) {
      stringbuffer.append(outputs(i))
      stringbuffer.append(" ")
      stringbuffer.append(inputs(i).map({case (x,y) => x + ":" + y}).mkString(" "))
      stringbuffer.append("\n")
    }

    writer.write(stringbuffer.toString())
    writer.close()

    writerp.write((1 to parameters.size).zip(parameters).map({case (x,y) => x + ":" + y}).mkString(" "))
    writerp.close()
  }

  def computeL(filename: String): Double = {
    val lines = scala.io.Source.fromFile(filename).getLines()
    var max = 0d
    while (lines.hasNext) {
      val line = lines.next
      val values = line.split(" ").drop(1)
      val thing = values.map(x => x.split(":")(1).toDouble)
      max = Math.max(max, thing.map(x => x * x).sum)
    }
    max
  }
}
