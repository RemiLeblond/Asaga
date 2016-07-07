import com.google.common.util.concurrent.AtomicDoubleArray

/**
 * Created by rleblond on 1/14/16.
 */
package object utils {

  /**
    *  Sparse CSR matrix format.
    * See http://netlib.org/linalg/html_templates/node91.html
    */
  case class CSRMatrix(data: Array[Double], indices: Array[Int], indPtr: Array[Int], nRows: Int, nCols: Int)

  def loadDataCSR(filename: String, nRows: Int, nCols: Int, nNonzero: Int): (CSRMatrix, Array[Double]) = {
    val lines = scala.io.Source.fromFile(filename).getLines()
    val data = Array.fill[Double](nNonzero)(0d)
    val indices = Array.fill[Int](nNonzero)(0)
    val indPtr = Array.fill[Int](nRows + 1)(0)
    val output = Array.fill[Double](nRows)(0)
    var counter = 0  // counter of nonzero elements
    var i = 0  // counter of rows

    // Using an iterator allows for bigger datasets
    while (lines.hasNext) {
      val line = lines.next
      val values = line.split(" ")
      output(i) = values(0).toDouble
      for (j <- 1 until values.length) {
        val Array(index, value) = values(j).split(":")
        data(counter) = value.toDouble
        indices(counter) = index.toInt - 1
        counter += 1
      }
      indPtr(i + 1) = counter
      i += 1
    }

    (CSRMatrix(data, indices, indPtr, nRows, nCols), output)
  }

  def loadDataRealSim(): (CSRMatrix, Array[Double]) = {
    val nRows = 72309
    val nCols = 20958
    val nNonzero = 3709083
    val data_dir = sys.env("SAGA_DATA_DIR")
    loadDataCSR(data_dir + "/real-sim", nRows, nCols, nNonzero)
  }

  def loadDataAlpha(): (CSRMatrix, Array[Double]) = {
    val nRows = 500000
    val nCols = 500
    val nNonzero = 250000000
    val data_dir = sys.env("SAGA_DATA_DIR")
    loadDataCSR(data_dir + "/alpha_train.libsvm", nRows, nCols, nNonzero)
  }

  def loadDataURL(): (CSRMatrix, Array[Double]) = {
    val nRows = 2396130
    val nCols = 3231961
    val nNonzero = 277058644
    val data_dir = sys.env("SAGA_DATA_DIR")
    loadDataCSR(data_dir + "/url_combined", nRows, nCols, nNonzero)
  }

  def loadDataRCV1(): (CSRMatrix, Array[Double]) = {
    val nRows = 20242
    val nCols = 47236
    val nNonzero = 1498952  // computed beforehand for simplicity
    val data_dir = sys.env("SAGA_DATA_DIR")
    loadDataCSR(data_dir + "/rcv1_train.binary", nRows, nCols, nNonzero)
  }

  def loadDataRCV1Test(): (CSRMatrix, Array[Double]) = {
    val data_dir = sys.env("SAGA_DATA_DIR")
    val nRows = 677399
    val nCols = 47236
    val nNonzero = 49556258
    loadDataCSR(data_dir + "/rcv1_test.binary", nRows, nCols, nNonzero)
  }

  def loadDataRCV1Full(): (CSRMatrix, Array[Double]) = {
    val data_dir = sys.env("SAGA_DATA_DIR")
    val nRows = 697641
    val nCols = 47236
    val nNonzero = 51055210
    loadDataCSR(data_dir + "/rcv1_full.binary", nRows, nCols, nNonzero)
  }

  def loadDataCovtype(): (CSRMatrix, Array[Double]) = {
    val data_dir = sys.env("SAGA_DATA_DIR")
    val nRows = 581012
    val nCols = 54
    val nNonzero = 31374648
    loadDataCSR(data_dir + "/covtype.libsvm.binary", nRows, nCols, nNonzero)
  }

  def loadToyDataset(): (CSRMatrix, Array[Double]) = {
    val data_dir = sys.env("SAGA_DATA_DIR")
    val nRows = 1000
    val nCols = 500
    val nNonzero = 500000
    loadDataCSR(data_dir + "/testdata", nRows, nCols, nNonzero)
  }

  // compute the probability that a given dimension is touched at each time step
  def computeInversePis(inputs: CSRMatrix, dimension: Int): Array[Double] = {
    val n = inputs.nRows
    val ipis = Array.fill[Double](dimension + 1)(0d)
    for (i <- 0 until n; j <- inputs.indPtr(i) until inputs.indPtr(i + 1)) {
      val k = inputs.indices(j)
      ipis(k) += 1
    }
    ipis.map(_ / n).map(1 / _)
  }

  // compute the ideal split of data for parallel batch gradient computation
  def splitCSR(inputs: CSRMatrix, cores: Int): Array[Int] = {
    val split = Array.fill[Int](cores + 1)(0)
    val splitSize = inputs.data.length / cores
    var k = 1
    for (i <- 0 until inputs.nRows) {
      if (inputs.indPtr(i) < k * splitSize && inputs.indPtr(i + 1) >= k * splitSize) {
        split(k) = i + 1
        k += 1
      }
    }

    // There may be up to (core - 1) more examples in the last split, depending on data.length % cores
    split(cores) = inputs.nRows

    split
  }

  def innerProduct(inputs: CSRMatrix, parameters: Array[Double], nRow: Int): Double = {
    (for (j <- inputs.indPtr(nRow) until inputs.indPtr(nRow + 1);
         k = inputs.indices(j);
         v = inputs.data(j))
      yield {parameters(k) * v}).sum
  }

  def atomicInnerProduct(inputs: CSRMatrix, atomicParameters: AtomicDoubleArray, nRow: Int): Double = {
    (for (j <- inputs.indPtr(nRow) until inputs.indPtr(nRow + 1);
          k = inputs.indices(j);
          v = inputs.data(j))
      yield atomicParameters.get(k) * v).sum
  }
}
