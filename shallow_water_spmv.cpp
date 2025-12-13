#include <Kokkos_Core.hpp>

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace swe_spmv {

using ExecSpace = Kokkos::DefaultExecutionSpace;
using MemSpace = ExecSpace::memory_space;
using RangePolicy = Kokkos::RangePolicy<ExecSpace>;

using IntView = Kokkos::View<int64_t*, MemSpace>;
using ConstIntView = Kokkos::View<const int64_t*, MemSpace>;
using DoubleView = Kokkos::View<double*, MemSpace>;
using ConstDoubleView = Kokkos::View<const double*, MemSpace>;

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

template <typename ViewType>
void copy_vector_into_view(
    const std::vector<typename ViewType::non_const_value_type>& host, ViewType view) {
  if (static_cast<typename ViewType::size_type>(host.size()) != view.extent(0)) {
    throw std::runtime_error("Host/view extent mismatch for " + std::string(view.label()));
  }

  auto mirror = Kokkos::create_mirror_view(view);
  for (size_t i = 0; i < host.size(); ++i) {
    mirror(static_cast<int>(i)) = host[i];
  }
  Kokkos::deep_copy(view, mirror);
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

class SparseMatVec {
 public:
  explicit SparseMatVec(const DiskArrays& arrays)
      : nc_(static_cast<int>(arrays.h.size())),
        ne_(static_cast<int>(arrays.src.size())),
        src_("src", ne_),
        dst_("dst", ne_),
        sx_("sx", ne_),
        vector_("h", nc_),
        matvec_("matvec", nc_),
        face_values_("face_values", ne_) {
    if (nc_ <= 0 || ne_ <= 0) {
      throw std::runtime_error("Mesh must contain at least one cell and edge");
    }

    copy_vector_into_view(arrays.src, src_);
    copy_vector_into_view(arrays.dst, dst_);
    copy_vector_into_view(arrays.sx, sx_);
    copy_vector_into_view(arrays.h, vector_);
    Kokkos::deep_copy(matvec_, 0.0);
    Kokkos::deep_copy(face_values_, 0.0);
  }

  int cells() const { return nc_; }
  int edges() const { return ne_; }

  void apply_once() {
    auto src = src_;
    auto dst = dst_;
    auto sx = sx_;
    auto vector = vector_;
    auto faces = face_values_;
    auto matvec = matvec_;

    Kokkos::parallel_for(
        "spmv_face_reconstruct", RangePolicy(0, ne_), KOKKOS_LAMBDA(const int e) {
          const auto source = static_cast<int>(src(e));
          faces(e) = vector(source);
        });

    Kokkos::deep_copy(matvec_, 0.0);

    Kokkos::parallel_for(
        "spmv_scatter", RangePolicy(0, ne_), KOKKOS_LAMBDA(const int e) {
          const auto cell = static_cast<int>(dst(e));
          const double contrib = faces(e) * sx(e);
          Kokkos::atomic_add(&matvec(cell), contrib);
        });

    Kokkos::parallel_for(
        "spmv_accumulate", RangePolicy(0, nc_), KOKKOS_LAMBDA(const int i) {
          vector(i) += matvec(i);
        });
  }

  double benchmark(int warmup_iters, int benchmark_iters) {
    if (warmup_iters < 0) {
      throw std::runtime_error("warmup iterations must be non-negative");
    }
    if (benchmark_iters <= 0) {
      throw std::runtime_error("benchmark iterations must be positive");
    }

    for (int i = 0; i < warmup_iters; ++i) {
      apply_once();
    }
    Kokkos::fence();

    const auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < benchmark_iters; ++i) {
      apply_once();
    }
    Kokkos::fence();
    const auto end = std::chrono::steady_clock::now();

    std::chrono::duration<double> seconds = end - start;
    return seconds.count();
  }

  void write_csv(const fs::path& path) const {
    auto host_vector = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), vector_);
    auto host_matvec = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), matvec_);

    fs::create_directories(path.parent_path());
    std::ofstream output(path);
    if (!output) {
      throw std::runtime_error("Failed to open " + path.string() + " for writing");
    }

    output << "cell,h,matvec\n";
    const auto n = host_vector.extent(0);
    for (size_t i = 0; i < n; ++i) {
      output << i << "," << host_vector(i) << "," << host_matvec(i) << "\n";
    }
  }

 private:
  int nc_{0};
  int ne_{0};

  IntView src_;
  IntView dst_;
  DoubleView sx_;
  DoubleView vector_;
  DoubleView matvec_;
  DoubleView face_values_;
};

struct Options {
  fs::path data_dir;
  int warmup_iters = 20;
  int benchmark_iters = 200;
  bool write_csv = false;
  fs::path csv_path;
};

void print_usage(const char* prog) {
  std::cout << "Usage: " << prog
            << " --data <directory> [--warmup N] [--iterations N] "
               "[--save-csv <path>]\n";
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
    } else if (arg == "--save-csv" && i + 1 < argc) {
      opts.csv_path = argv[++i];
      opts.write_csv = true;
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

}  // namespace swe_spmv

int main(int argc, char* argv[]) {
  Kokkos::ScopeGuard guard(argc, argv);

  swe_spmv::Options options;
  try {
    options = swe_spmv::parse_args(argc, argv);
  } catch (const std::exception& ex) {
    std::cerr << "Argument error: " << ex.what() << "\n";
    swe_spmv::print_usage(argv[0]);
    return 1;
  }

  try {
    auto arrays = swe_spmv::load_disk_arrays(options.data_dir);
    swe_spmv::SparseMatVec spmv(arrays);

    std::cout << "Loaded mesh with " << spmv.cells() << " cells and " << spmv.edges()
              << " edges.\n";

    const double seconds = spmv.benchmark(options.warmup_iters, options.benchmark_iters);
    const double ms_per_iter = seconds / options.benchmark_iters * 1000.0;

    std::cout << "SpMV ran " << options.benchmark_iters << " iterations in " << seconds
              << " seconds (" << ms_per_iter << " ms/iter).\n";
    std::cout << "Execution space: " << swe_spmv::ExecSpace::name() << "\n";

    if (options.write_csv) {
      spmv.write_csv(options.csv_path);
      std::cout << "Saved accumulated vector and latest matvec to "
                << options.csv_path.string() << "\n";
    }
  } catch (const std::exception& ex) {
    std::cerr << "Runtime error: " << ex.what() << "\n";
    return 1;
  }

  return 0;
}

