#include <cuda_runtime.h>
#include <cusparse.h>
#include <cub/device/device_spmv.cuh>

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace swe_cublas_spmv {

namespace fs = std::filesystem;

template <typename T>
std::vector<T> read_binary_vector(const fs::path& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    throw std::runtime_error("Failed to open " + path.string());
  }

  uint64_t count = 0;
  input.read(reinterpret_cast<char*>(&count), sizeof(uint64_t));
  if (!input) {
    throw std::runtime_error("Failed to read element count from " + path.string());
  }

  std::vector<T> data(count);
  input.read(reinterpret_cast<char*>(data.data()),
             static_cast<std::streamsize>(sizeof(T) * count));
  if (!input) {
    throw std::runtime_error("Failed to read payload from " + path.string());
  }

  return data;
}

struct DiskArrays {
  std::vector<int64_t> src;
  std::vector<int64_t> dst;
  std::vector<double> sx;
  std::vector<double> h;
};

DiskArrays load_disk_arrays(const fs::path& data_dir) {
  DiskArrays arrays;

  arrays.src = read_binary_vector<int64_t>(data_dir / "src.int64.bin");
  arrays.dst = read_binary_vector<int64_t>(data_dir / "dst.int64.bin");
  arrays.sx = read_binary_vector<double>(data_dir / "sx.float64.bin");
  arrays.h = read_binary_vector<double>(data_dir / "h.float64.bin");

  if (arrays.src.size() != arrays.dst.size()) {
    throw std::runtime_error("src and dst must have identical sizes");
  }
  if (arrays.sx.size() != arrays.src.size()) {
    throw std::runtime_error("sx must match number of edges");
  }
  if (arrays.h.empty()) {
    throw std::runtime_error("h must be present and non-empty");
  }

  return arrays;
}

struct MatrixData {
  int rows = 0;
  int cols = 0;
  int nnz = 0;
  std::vector<int> coo_rows;
  std::vector<int> coo_cols;
  std::vector<double> coo_values;
  std::vector<int> csr_row_ptr;
  std::vector<int> csr_col_idx;
  std::vector<double> csr_values;
  std::vector<double> initial_vector;
};

int64_t require_nonnegative(int64_t value, const std::string& label) {
  if (value < 0) {
    throw std::runtime_error(label + " must be non-negative");
  }
  return value;
}

MatrixData build_matrix(const DiskArrays& arrays) {
  MatrixData matrix;

  const auto max_int = std::numeric_limits<int>::max();
  const auto nnz64 = require_nonnegative(static_cast<int64_t>(arrays.src.size()), "nnz count");
  const auto rows64 = require_nonnegative(static_cast<int64_t>(arrays.h.size()), "cell count");
    if (nnz64 > max_int || rows64 > max_int) {
      throw std::runtime_error("Matrix dimensions exceed 32-bit index limits required by cuSPARSE");
  }

  matrix.rows = static_cast<int>(rows64);
  matrix.cols = matrix.rows;
  matrix.nnz = static_cast<int>(nnz64);
  matrix.initial_vector = arrays.h;

  matrix.csr_row_ptr.assign(matrix.rows + 1, 0);
  for (int i = 0; i < matrix.nnz; ++i) {
    const auto row64 = arrays.dst[static_cast<size_t>(i)];
    if (row64 < 0 || row64 > max_int) {
      throw std::runtime_error("dst index exceeds 32-bit range near entry " + std::to_string(i));
    }
    const int row = static_cast<int>(row64);
    ++matrix.csr_row_ptr[static_cast<size_t>(row) + 1];
  }
  for (int i = 0; i < matrix.rows; ++i) {
    matrix.csr_row_ptr[i + 1] += matrix.csr_row_ptr[i];
  }

  matrix.csr_col_idx.assign(matrix.nnz, 0);
  matrix.csr_values.assign(matrix.nnz, 0.0);
  std::vector<int> fill_offset(matrix.rows, 0);
  for (int i = 0; i < matrix.nnz; ++i) {
    const auto row64 = arrays.dst[static_cast<size_t>(i)];
    const auto col64 = arrays.src[static_cast<size_t>(i)];
    if (col64 < 0 || col64 > max_int) {
      throw std::runtime_error("src index exceeds 32-bit range near entry " + std::to_string(i));
    }
    const int row = static_cast<int>(row64);
    const int col = static_cast<int>(col64);
    const int slot = matrix.csr_row_ptr[row] + fill_offset[row];
    fill_offset[row] += 1;
    matrix.csr_col_idx[slot] = col;
    matrix.csr_values[slot] = arrays.sx[static_cast<size_t>(i)];
  }

  matrix.coo_rows.assign(matrix.nnz, 0);
  matrix.coo_cols.assign(matrix.nnz, 0);
  matrix.coo_values.assign(matrix.nnz, 0.0);
  int position = 0;
  for (int row = 0; row < matrix.rows; ++row) {
    for (int idx = matrix.csr_row_ptr[row]; idx < matrix.csr_row_ptr[row + 1]; ++idx) {
      matrix.coo_rows[position] = row;
      matrix.coo_cols[position] = matrix.csr_col_idx[idx];
      matrix.coo_values[position] = matrix.csr_values[idx];
      ++position;
    }
  }

  return matrix;
}

std::string cusparse_status_to_string(cusparseStatus_t status) {
  switch (status) {
    case CUSPARSE_STATUS_SUCCESS:
      return "SUCCESS";
    case CUSPARSE_STATUS_NOT_INITIALIZED:
      return "NOT_INITIALIZED";
    case CUSPARSE_STATUS_ALLOC_FAILED:
      return "ALLOC_FAILED";
    case CUSPARSE_STATUS_INVALID_VALUE:
      return "INVALID_VALUE";
    case CUSPARSE_STATUS_ARCH_MISMATCH:
      return "ARCH_MISMATCH";
    case CUSPARSE_STATUS_MAPPING_ERROR:
      return "MAPPING_ERROR";
    case CUSPARSE_STATUS_EXECUTION_FAILED:
      return "EXECUTION_FAILED";
    case CUSPARSE_STATUS_INTERNAL_ERROR:
      return "INTERNAL_ERROR";
    case CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED:
      return "MATRIX_TYPE_NOT_SUPPORTED";
    default:
      return "UNKNOWN";
  }
}

#define CUDA_CHECK(expr)                                                                  \
  do {                                                                                    \
    cudaError_t _err = (expr);                                                            \
    if (_err != cudaSuccess) {                                                            \
      throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(_err));   \
    }                                                                                     \
  } while (0)

#define CUB_CHECK(expr)                                                                   \
  do {                                                                                    \
    cudaError_t _err = (expr);                                                            \
    if (_err != cudaSuccess) {                                                            \
      throw std::runtime_error(std::string("CUB error: ") + cudaGetErrorString(_err));    \
    }                                                                                     \
  } while (0)

#define CUSPARSE_CHECK(expr)                                                                  \
  do {                                                                                        \
    cusparseStatus_t _status = (expr);                                                         \
    if (_status != CUSPARSE_STATUS_SUCCESS) {                                                 \
      throw std::runtime_error("cuSPARSE error: " + cusparse_status_to_string(_status));      \
    }                                                                                         \
  } while (0)

class DeviceMatrix {
 public:
  explicit DeviceMatrix(const MatrixData& host) : rows_(host.rows), cols_(host.cols), nnz_(host.nnz) {
    if (rows_ <= 0 || cols_ <= 0 || nnz_ <= 0) {
      throw std::runtime_error("Matrix must contain positive dimensions");
    }

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_coo_rows_), sizeof(int) * nnz_));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_coo_cols_), sizeof(int) * nnz_));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_values_), sizeof(double) * nnz_));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_csr_row_ptr_), sizeof(int) * (rows_ + 1)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_csr_col_idx_), sizeof(int) * nnz_));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_csr_values_), sizeof(double) * nnz_));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_input_vector_), sizeof(double) * cols_));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_output_vector_), sizeof(double) * rows_));

    CUDA_CHECK(cudaMemcpy(d_coo_rows_, host.coo_rows.data(), sizeof(int) * nnz_,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_coo_cols_, host.coo_cols.data(), sizeof(int) * nnz_,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_values_, host.coo_values.data(), sizeof(double) * nnz_,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_csr_row_ptr_, host.csr_row_ptr.data(), sizeof(int) * (rows_ + 1),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_csr_col_idx_, host.csr_col_idx.data(), sizeof(int) * nnz_,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_csr_values_, host.csr_values.data(), sizeof(double) * nnz_,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_input_vector_, host.initial_vector.data(), sizeof(double) * cols_,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_output_vector_, 0, sizeof(double) * rows_));
  }

  ~DeviceMatrix() {
    cudaFree(d_coo_rows_);
    cudaFree(d_coo_cols_);
    cudaFree(d_values_);
    cudaFree(d_csr_row_ptr_);
    cudaFree(d_csr_col_idx_);
    cudaFree(d_csr_values_);
    cudaFree(d_input_vector_);
    cudaFree(d_output_vector_);
  }

  DeviceMatrix(const DeviceMatrix&) = delete;
  DeviceMatrix& operator=(const DeviceMatrix&) = delete;
  DeviceMatrix(DeviceMatrix&&) = delete;
  DeviceMatrix& operator=(DeviceMatrix&&) = delete;

  int rows() const { return rows_; }
  int cols() const { return cols_; }
  int nnz() const { return nnz_; }

  int* coo_rows() const { return d_coo_rows_; }
  int* coo_cols() const { return d_coo_cols_; }
  double* coo_values() const { return d_values_; }

  int* csr_row_ptr() const { return d_csr_row_ptr_; }
  int* csr_col_idx() const { return d_csr_col_idx_; }
  double* csr_values() const { return d_csr_values_; }

  double* input_vector() const { return d_input_vector_; }
  double* output_vector() const { return d_output_vector_; }

  void reset_output() const {
    CUDA_CHECK(cudaMemset(d_output_vector_, 0, sizeof(double) * rows_));
  }

 private:
  int rows_{0};
  int cols_{0};
  int nnz_{0};

  int* d_coo_rows_{nullptr};
  int* d_coo_cols_{nullptr};
  double* d_values_{nullptr};
  int* d_csr_row_ptr_{nullptr};
  int* d_csr_col_idx_{nullptr};
  double* d_csr_values_{nullptr};
  double* d_input_vector_{nullptr};
  double* d_output_vector_{nullptr};
};

class SpmvBenchmark {
 public:
  explicit SpmvBenchmark(const MatrixData& host_matrix) : device_(host_matrix) {
    CUSPARSE_CHECK(cusparseCreate(&cusparse_handle_));
    CUSPARSE_CHECK(cusparseSetPointerMode(cusparse_handle_, CUSPARSE_POINTER_MODE_HOST));

    CUSPARSE_CHECK(cusparseCreateCoo(&coo_descr_, device_.rows(), device_.cols(), device_.nnz(),
                                     device_.coo_rows(), device_.coo_cols(), device_.coo_values(),
                                     CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F));
    CUSPARSE_CHECK(cusparseCreateCsr(&csr_descr_, device_.rows(), device_.cols(), device_.nnz(),
                                     device_.csr_row_ptr(), device_.csr_col_idx(),
                                     device_.csr_values(), CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                     CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F));
    CUSPARSE_CHECK(cusparseCreateDnVec(&x_vec_, device_.cols(), device_.input_vector(), CUDA_R_64F));
    CUSPARSE_CHECK(cusparseCreateDnVec(&y_vec_, device_.rows(), device_.output_vector(), CUDA_R_64F));

    double alpha = 1.0;
    double beta = 0.0;
    size_t coo_buffer_size = 0;
    size_t csr_buffer_size = 0;
    CUSPARSE_CHECK(cusparseSpMV_bufferSize(
        cusparse_handle_, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, coo_descr_, x_vec_, &beta,
        y_vec_, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, &coo_buffer_size));
    CUSPARSE_CHECK(cusparseSpMV_bufferSize(
        cusparse_handle_, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, csr_descr_, x_vec_, &beta,
        y_vec_, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, &csr_buffer_size));
    const size_t buffer_size = std::max(coo_buffer_size, csr_buffer_size);
    CUDA_CHECK(cudaMalloc(&spmv_buffer_, buffer_size));

    auto cub_status = cub::DeviceSpmv::CsrMV(nullptr, cub_buffer_bytes_, device_.csr_values(),
                                             device_.csr_row_ptr(), device_.csr_col_idx(),
                                             device_.input_vector(), device_.output_vector(),
                                             device_.rows(), device_.cols(), device_.nnz());
    if (cub_status == cudaSuccess) {
      CUDA_CHECK(cudaMalloc(&cub_buffer_, cub_buffer_bytes_));
      cub_available_ = true;
    } else if (cub_status == cudaErrorNoKernelImageForDevice) {
      cub_available_ = false;
      cub_error_message_ = "cudaErrorNoKernelImageForDevice";
      cub_buffer_bytes_ = 0;
    } else {
      CUB_CHECK(cub_status);
    }
  }

  ~SpmvBenchmark() {
    cudaFree(spmv_buffer_);
    if (cub_buffer_) {
      cudaFree(cub_buffer_);
    }
    cusparseDestroySpMat(coo_descr_);
    cusparseDestroySpMat(csr_descr_);
    cusparseDestroyDnVec(x_vec_);
    cusparseDestroyDnVec(y_vec_);
    cusparseDestroy(cusparse_handle_);
  }

  SpmvBenchmark(const SpmvBenchmark&) = delete;
  SpmvBenchmark& operator=(const SpmvBenchmark&) = delete;
  SpmvBenchmark(SpmvBenchmark&&) = delete;
  SpmvBenchmark& operator=(SpmvBenchmark&&) = delete;

  double benchmark_coo(int warmup_iters, int benchmark_iters) {
    if (benchmark_iters <= 0) {
      throw std::runtime_error("benchmark iterations must be positive");
    }
    if (warmup_iters < 0) {
      throw std::runtime_error("warmup iterations must be non-negative");
    }

    const double alpha = 1.0;
    const double beta = 0.0;
    device_.reset_output();

    for (int i = 0; i < warmup_iters; ++i) {
      CUSPARSE_CHECK(cusparseSpMV(cusparse_handle_, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha,
                                  coo_descr_, x_vec_, &beta, y_vec_, CUDA_R_64F,
                                  CUSPARSE_SPMV_ALG_DEFAULT, spmv_buffer_));
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < benchmark_iters; ++i) {
      CUSPARSE_CHECK(cusparseSpMV(cusparse_handle_, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha,
                                  coo_descr_, x_vec_, &beta, y_vec_, CUDA_R_64F,
                                  CUSPARSE_SPMV_ALG_DEFAULT, spmv_buffer_));
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float elapsed_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return static_cast<double>(elapsed_ms) / 1000.0;
  }

  double benchmark_csr(int warmup_iters, int benchmark_iters) {
    if (benchmark_iters <= 0) {
      throw std::runtime_error("benchmark iterations must be positive");
    }
    if (warmup_iters < 0) {
      throw std::runtime_error("warmup iterations must be non-negative");
    }

    const double alpha = 1.0;
    const double beta = 0.0;
    device_.reset_output();

    for (int i = 0; i < warmup_iters; ++i) {
      CUSPARSE_CHECK(cusparseSpMV(cusparse_handle_, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha,
                                  csr_descr_, x_vec_, &beta, y_vec_, CUDA_R_64F,
                                  CUSPARSE_SPMV_ALG_DEFAULT, spmv_buffer_));
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < benchmark_iters; ++i) {
      CUSPARSE_CHECK(cusparseSpMV(cusparse_handle_, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha,
                                  csr_descr_, x_vec_, &beta, y_vec_, CUDA_R_64F,
                                  CUSPARSE_SPMV_ALG_DEFAULT, spmv_buffer_));
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float elapsed_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return static_cast<double>(elapsed_ms) / 1000.0;
  }

  double benchmark_cub_csr(int warmup_iters, int benchmark_iters) {
    if (!cub_available_) {
      throw std::runtime_error("CUB CSR benchmark unavailable: " + cub_error_message_);
    }
    if (benchmark_iters <= 0) {
      throw std::runtime_error("benchmark iterations must be positive");
    }
    if (warmup_iters < 0) {
      throw std::runtime_error("warmup iterations must be non-negative");
    }

    device_.reset_output();

    for (int i = 0; i < warmup_iters; ++i) {
      CUB_CHECK(cub::DeviceSpmv::CsrMV(cub_buffer_, cub_buffer_bytes_, device_.csr_values(),
                                       device_.csr_row_ptr(), device_.csr_col_idx(),
                                       device_.input_vector(), device_.output_vector(),
                                       device_.rows(), device_.cols(), device_.nnz()));
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < benchmark_iters; ++i) {
      CUB_CHECK(cub::DeviceSpmv::CsrMV(cub_buffer_, cub_buffer_bytes_, device_.csr_values(),
                                       device_.csr_row_ptr(), device_.csr_col_idx(),
                                       device_.input_vector(), device_.output_vector(),
                                       device_.rows(), device_.cols(), device_.nnz()));
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float elapsed_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return static_cast<double>(elapsed_ms) / 1000.0;
  }

  int rows() const { return device_.rows(); }
  int nnz() const { return device_.nnz(); }
  bool cub_available() const { return cub_available_; }
  const std::string& cub_status() const { return cub_error_message_; }

 private:
  DeviceMatrix device_;
  cusparseHandle_t cusparse_handle_{};
  cusparseSpMatDescr_t coo_descr_{};
  cusparseSpMatDescr_t csr_descr_{};
  cusparseDnVecDescr_t x_vec_{};
  cusparseDnVecDescr_t y_vec_{};
  void* spmv_buffer_{nullptr};
  size_t cub_buffer_bytes_{0};
  void* cub_buffer_{nullptr};
  bool cub_available_{false};
  std::string cub_error_message_ = "not initialized";
};

struct Options {
  fs::path data_dir;
  int warmup_iters = 20;
  int benchmark_iters = 200;
};

void print_usage(const char* prog) {
  std::cout << "Usage: " << prog << " --data <directory> [--warmup N] [--iterations N]\n";
}

Options parse_args(int argc, char* argv[]) {
  Options opts;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--help" || arg == "-h") {
      print_usage(argv[0]);
      std::exit(0);
    } else if (arg == "--data" && i + 1 < argc) {
      opts.data_dir = argv[++i];
    } else if (arg == "--warmup" && i + 1 < argc) {
      opts.warmup_iters = std::stoi(argv[++i]);
    } else if (arg == "--iterations" && i + 1 < argc) {
      opts.benchmark_iters = std::stoi(argv[++i]);
    } else {
      std::cerr << "Unknown or incomplete argument: " << arg << "\n";
      print_usage(argv[0]);
      std::exit(1);
    }
  }

  if (opts.data_dir.empty()) {
    throw std::runtime_error("--data <directory> is required");
  }
  if (!fs::exists(opts.data_dir)) {
    throw std::runtime_error("Data directory does not exist: " + opts.data_dir.string());
  }
  if (opts.warmup_iters < 0) {
    throw std::runtime_error("--warmup must be non-negative");
  }
  if (opts.benchmark_iters <= 0) {
    throw std::runtime_error("--iterations must be positive");
  }

  return opts;
}

}  // namespace swe_cublas_spmv

int main(int argc, char* argv[]) {
  swe_cublas_spmv::Options options;
  try {
    options = swe_cublas_spmv::parse_args(argc, argv);
  } catch (const std::exception& ex) {
    std::cerr << "Argument error: " << ex.what() << "\n";
    swe_cublas_spmv::print_usage(argv[0]);
    return 1;
  }

  try {
    auto arrays = swe_cublas_spmv::load_disk_arrays(options.data_dir);
    auto matrix = swe_cublas_spmv::build_matrix(arrays);
    swe_cublas_spmv::SpmvBenchmark benchmark(matrix);

    std::cout << "Loaded matrix with " << benchmark.rows() << " rows and " << benchmark.nnz()
              << " nonzeros.\n";
    std::cout << "Running cuSPARSE COO benchmark...\n";
    const double coo_seconds =
        benchmark.benchmark_coo(options.warmup_iters, options.benchmark_iters);
    const double coo_ms_per_iter = coo_seconds / options.benchmark_iters * 1000.0;
    std::cout << "cuSPARSE COO: " << options.benchmark_iters << " iterations in " << coo_seconds
              << " seconds (" << coo_ms_per_iter << " ms/iter)\n";

    std::cout << "Running cuSPARSE CSR benchmark...\n";
    const double csr_seconds =
        benchmark.benchmark_csr(options.warmup_iters, options.benchmark_iters);
    const double csr_ms_per_iter = csr_seconds / options.benchmark_iters * 1000.0;
    std::cout << "cuSPARSE CSR: " << options.benchmark_iters << " iterations in " << csr_seconds
              << " seconds (" << csr_ms_per_iter << " ms/iter)\n";

    if (benchmark.cub_available()) {
      std::cout << "Running CUB CSR benchmark...\n";
      const double cub_seconds =
          benchmark.benchmark_cub_csr(options.warmup_iters, options.benchmark_iters);
      const double cub_ms_per_iter = cub_seconds / options.benchmark_iters * 1000.0;
      std::cout << "CUB CSR: " << options.benchmark_iters << " iterations in " << cub_seconds
                << " seconds (" << cub_ms_per_iter << " ms/iter)\n";
    } else {
      std::cout << "Skipping CUB CSR benchmark (" << benchmark.cub_status() << ")\n";
    }
  } catch (const std::exception& ex) {
    std::cerr << "Runtime error: " << ex.what() << "\n";
    return 1;
  }

  return 0;
}
