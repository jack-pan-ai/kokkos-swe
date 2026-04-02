#include <Kokkos_Core.hpp>

#include <array>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace swe {

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
  std::vector<int64_t> bcells;

  std::vector<double> alpha;
  std::vector<double> area;
  std::vector<double> sx;
  std::vector<double> sy;
  std::vector<double> bsx;
  std::vector<double> bsy;
  std::vector<double> h;
  std::vector<double> x;
  std::vector<double> y;
};

DiskArrays load_disk_arrays(const fs::path& data_dir) {
  DiskArrays arrays;

  arrays.src = read_binary_vector<int64_t>(data_dir / "src.int64.bin");
  arrays.dst = read_binary_vector<int64_t>(data_dir / "dst.int64.bin");
  arrays.bcells = read_binary_vector<int64_t>(data_dir / "bcells.int64.bin");

  arrays.alpha = read_binary_vector<double>(data_dir / "alpha.float64.bin");
  arrays.area = read_binary_vector<double>(data_dir / "area.float64.bin");
  arrays.sx = read_binary_vector<double>(data_dir / "sx.float64.bin");
  arrays.sy = read_binary_vector<double>(data_dir / "sy.float64.bin");
  arrays.bsx = read_binary_vector<double>(data_dir / "bsx.float64.bin");
  arrays.bsy = read_binary_vector<double>(data_dir / "bsy.float64.bin");
  arrays.h = read_binary_vector<double>(data_dir / "h.float64.bin");
  arrays.x = read_binary_vector<double>(data_dir / "x.float64.bin");
  arrays.y = read_binary_vector<double>(data_dir / "y.float64.bin");

  if (arrays.src.size() != arrays.dst.size()) {
    throw std::runtime_error("src and dst must have identical sizes");
  }
  if (arrays.alpha.size() != arrays.src.size()) {
    throw std::runtime_error("alpha size must match number of edges");
  }
  if (arrays.sx.size() != arrays.src.size() || arrays.sy.size() != arrays.src.size()) {
    throw std::runtime_error("Metric arrays must match number of edges");
  }
  if (arrays.area.size() != arrays.h.size()) {
    throw std::runtime_error("area must match number of cells");
  }
  if (!arrays.bcells.empty()) {
    if (arrays.bsx.size() != arrays.bcells.size() || arrays.bsy.size() != arrays.bcells.size()) {
      throw std::runtime_error("Boundary metric arrays must match number of boundary cells");
    }
  }
  if (arrays.x.size() != arrays.h.size() || arrays.y.size() != arrays.h.size()) {
    throw std::runtime_error("x/y coordinates must match number of cells");
  }

  return arrays;
}

struct StageViews {
  DoubleView h;
  DoubleView uh;
  DoubleView vh;
};

class ShallowWaterEquationFused {
 public:
  explicit ShallowWaterEquationFused(const DiskArrays& arrays, double dt);
  void step();
  int cells() const { return nc_; }
  ConstDoubleView h() const { return h_; }
  ConstDoubleView x() const { return x_; }
  ConstDoubleView y() const { return y_; }

  void compute_delta(const ConstDoubleView& h_in,
                     const ConstDoubleView& uh_in,
                     const ConstDoubleView& vh_in,
                     const DoubleView& delta_h,
                     const DoubleView& delta_uh,
                     const DoubleView& delta_vh) const;

  void combine_state(const ConstDoubleView& base,
                     const ConstDoubleView& delta,
                     double scale,
                     const DoubleView& out) const {
    auto base_local = base;
    auto delta_local = delta;
    auto out_local = out;
    Kokkos::parallel_for("combine_state", RangePolicy(0, nc_), KOKKOS_LAMBDA(const int i) {
      out_local(i) = base_local(i) + scale * delta_local(i);
    });
  }

 private:
  double dt_;
  int nc_{0};
  int ne_{0};
  int nbc_{0};

  IntView src_;
  IntView dst_;
  IntView bcells_;

  DoubleView alpha_;
  DoubleView area_;
  DoubleView sx_;
  DoubleView sy_;
  DoubleView bsx_;
  DoubleView bsy_;

  DoubleView h_;
  DoubleView uh_;
  DoubleView vh_;
  DoubleView x_;
  DoubleView y_;

  std::array<StageViews, 4> stages_;
  DoubleView tmp_h_;
  DoubleView tmp_uh_;
  DoubleView tmp_vh_;
};

ShallowWaterEquationFused::ShallowWaterEquationFused(const DiskArrays& arrays, double dt)
    : dt_(dt),
      nc_(static_cast<int>(arrays.h.size())),
      ne_(static_cast<int>(arrays.src.size())),
      nbc_(static_cast<int>(arrays.bcells.size())),
      src_("src", ne_),
      dst_("dst", ne_),
      bcells_("bcells", nbc_),
      alpha_("alpha", ne_),
      area_("area", nc_),
      sx_("sx", ne_),
      sy_("sy", ne_),
      bsx_("bsx", nbc_),
      bsy_("bsy", nbc_),
      h_("h", nc_),
      uh_("uh", nc_),
      vh_("vh", nc_),
      x_("x", nc_),
      y_("y", nc_),
      tmp_h_("tmp_h", nc_),
      tmp_uh_("tmp_uh", nc_),
      tmp_vh_("tmp_vh", nc_) {
  if (dt_ <= 0.0) {
    throw std::runtime_error("dt must be positive");
  }

  copy_vector_into_view(arrays.src, src_);
  copy_vector_into_view(arrays.dst, dst_);
  if (nbc_ > 0) {
    copy_vector_into_view(arrays.bcells, bcells_);
  }

  copy_vector_into_view(arrays.alpha, alpha_);
  copy_vector_into_view(arrays.area, area_);
  copy_vector_into_view(arrays.sx, sx_);
  copy_vector_into_view(arrays.sy, sy_);
  if (nbc_ > 0) {
    copy_vector_into_view(arrays.bsx, bsx_);
    copy_vector_into_view(arrays.bsy, bsy_);
  }

  copy_vector_into_view(arrays.h, h_);
  Kokkos::deep_copy(uh_, 0.0);
  Kokkos::deep_copy(vh_, 0.0);
  copy_vector_into_view(arrays.x, x_);
  copy_vector_into_view(arrays.y, y_);

  for (int i = 0; i < 4; ++i) {
    const auto idx = std::to_string(i);
    stages_[i].h = DoubleView(Kokkos::view_alloc("delta_h_" + idx), nc_);
    stages_[i].uh = DoubleView(Kokkos::view_alloc("delta_uh_" + idx), nc_);
    stages_[i].vh = DoubleView(Kokkos::view_alloc("delta_vh_" + idx), nc_);
  }
}

void ShallowWaterEquationFused::compute_delta(const ConstDoubleView& h_in,
                                              const ConstDoubleView& uh_in,
                                              const ConstDoubleView& vh_in,
                                              const DoubleView& delta_h,
                                              const DoubleView& delta_uh,
                                              const DoubleView& delta_vh) const {
  Kokkos::deep_copy(delta_h, 0.0);
  Kokkos::deep_copy(delta_uh, 0.0);
  Kokkos::deep_copy(delta_vh, 0.0);

  auto src = src_;
  auto dst = dst_;
  auto alpha = alpha_;
  auto sx = sx_;
  auto sy = sy_;

  auto dh = delta_h;
  auto du = delta_uh;
  auto dv = delta_vh;

  // Fused edge pipeline: face reconstruction + velocity + flux + scatter.
  Kokkos::parallel_for("delta_edges_fused", RangePolicy(0, ne_), KOKKOS_LAMBDA(const int e) {
    const int left = static_cast<int>(src(e));
    const int right = static_cast<int>(dst(e));
    const double a = alpha(e);

    const double h_face = (1.0 - a) * h_in(left) + a * h_in(right);
    const double uh_face = (1.0 - a) * uh_in(left) + a * uh_in(right);
    const double vh_face = (1.0 - a) * vh_in(left) + a * vh_in(right);

    // NOTE: The EASIER tutorial uses direct division. (No epsilon-guard.)
    const double u = uh_face / h_face;
    const double v = vh_face / h_face;

    const double h_square = 0.5 * h_face * h_face;
    const double sx_val = sx(e);
    const double sy_val = sy(e);

    const double uh_sx = uh_face * sx_val;
    const double vh_sy = vh_face * sy_val;

    const double contrib_h = uh_sx + vh_sy;
    const double contrib_uh = (u * uh_face + h_square) * sx_val + u * vh_sy;
    const double contrib_vh = v * uh_sx + (v * vh_face + h_square) * sy_val;

    Kokkos::atomic_add(&dh(right), -contrib_h);
    Kokkos::atomic_add(&du(right), -contrib_uh);
    Kokkos::atomic_add(&dv(right), -contrib_vh);
  });

  if (nbc_ > 0) {
    auto bcells = bcells_;
    auto bsx = bsx_;
    auto bsy = bsy_;
    auto h_cells = h_in;

    Kokkos::parallel_for("delta_boundary", RangePolicy(0, nbc_), KOKKOS_LAMBDA(const int i) {
      const int cell = static_cast<int>(bcells(i));
      const double h_cell = h_cells(cell);
      const double h_square = 0.5 * h_cell * h_cell;
      Kokkos::atomic_add(&du(cell), -h_square * bsx(i));
      Kokkos::atomic_add(&dv(cell), -h_square * bsy(i));
    });
  }

  auto area = area_;
  Kokkos::parallel_for("delta_scale", RangePolicy(0, nc_), KOKKOS_LAMBDA(const int i) {
    const double inv_area = 1.0 / area(i);
    dh(i) *= inv_area;
    du(i) *= inv_area;
    dv(i) *= inv_area;
  });
}

void ShallowWaterEquationFused::step() {
  compute_delta(h_, uh_, vh_, stages_[0].h, stages_[0].uh, stages_[0].vh);

  const double half_dt = 0.5 * dt_;
  combine_state(h_, stages_[0].h, half_dt, tmp_h_);
  combine_state(uh_, stages_[0].uh, half_dt, tmp_uh_);
  combine_state(vh_, stages_[0].vh, half_dt, tmp_vh_);

  compute_delta(tmp_h_, tmp_uh_, tmp_vh_, stages_[1].h, stages_[1].uh, stages_[1].vh);

  combine_state(h_, stages_[1].h, half_dt, tmp_h_);
  combine_state(uh_, stages_[1].uh, half_dt, tmp_uh_);
  combine_state(vh_, stages_[1].vh, half_dt, tmp_vh_);

  compute_delta(tmp_h_, tmp_uh_, tmp_vh_, stages_[2].h, stages_[2].uh, stages_[2].vh);

  combine_state(h_, stages_[2].h, dt_, tmp_h_);
  combine_state(uh_, stages_[2].uh, dt_, tmp_uh_);
  combine_state(vh_, stages_[2].vh, dt_, tmp_vh_);

  compute_delta(tmp_h_, tmp_uh_, tmp_vh_, stages_[3].h, stages_[3].uh, stages_[3].vh);

  const double factor = dt_ / 6.0;
  auto h = h_;
  auto uh = uh_;
  auto vh = vh_;

  auto d1_h = stages_[0].h;
  auto d2_h = stages_[1].h;
  auto d3_h = stages_[2].h;
  auto d4_h = stages_[3].h;
  auto d1_uh = stages_[0].uh;
  auto d2_uh = stages_[1].uh;
  auto d3_uh = stages_[2].uh;
  auto d4_uh = stages_[3].uh;
  auto d1_vh = stages_[0].vh;
  auto d2_vh = stages_[1].vh;
  auto d3_vh = stages_[2].vh;
  auto d4_vh = stages_[3].vh;

  Kokkos::parallel_for("rk4_update", RangePolicy(0, nc_), KOKKOS_LAMBDA(const int i) {
    h(i) += factor * (d1_h(i) + d2_h(i) + d3_h(i) + d4_h(i));
    uh(i) += factor * (d1_uh(i) + d2_uh(i) + d3_uh(i) + d4_uh(i));
    vh(i) += factor * (d1_vh(i) + d2_vh(i) + d3_vh(i) + d4_vh(i));
  });
}

struct Options {
  fs::path data_dir;
  fs::path output_dir;
  int steps = 1000;
  int output_interval = 10;
  double dt = 0.0005;
  bool write_output = false;
  bool profile = false;
  int profile_warmup = 10;
  int profile_iterations = -1;
};

void print_usage(const char* prog) {
  std::cout << "Usage: " << prog
            << " --data <directory> [--steps N] [--dt value] "
               "[--output <directory>] [--output-interval M] "
               "[--profile] [--profile-iters N] [--profile-warmup N]\n";
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
    } else if (arg == "--steps" && i + 1 < argc) {
      opts.steps = std::stoi(argv[++i]);
    } else if (arg == "--dt" && i + 1 < argc) {
      opts.dt = std::stod(argv[++i]);
    } else if (arg == "--output" && i + 1 < argc) {
      opts.output_dir = argv[++i];
      opts.write_output = true;
    } else if (arg == "--output-interval" && i + 1 < argc) {
      opts.output_interval = std::stoi(argv[++i]);
    } else if (arg == "--profile") {
      opts.profile = true;
    } else if (arg == "--profile-iters" && i + 1 < argc) {
      opts.profile_iterations = std::stoi(argv[++i]);
    } else if (arg == "--profile-warmup" && i + 1 < argc) {
      opts.profile_warmup = std::stoi(argv[++i]);
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
  if (opts.steps <= 0) {
    throw std::runtime_error("--steps must be positive");
  }
  if (opts.output_interval <= 0) {
    throw std::runtime_error("--output-interval must be positive");
  }
  if (opts.profile_warmup < 0) {
    throw std::runtime_error("--profile-warmup must be non-negative");
  }
  if (opts.profile) {
    if (opts.profile_iterations <= 0) {
      const std::string exec_name = ExecSpace::name();
      const bool defaults_to_gpu = exec_name.find("Cuda") != std::string::npos ||
                                   exec_name.find("HIP") != std::string::npos;
      opts.profile_iterations = defaults_to_gpu ? 200 : 20;
    }
    if (opts.profile_iterations <= 0) {
      throw std::runtime_error("--profile-iters must be positive");
    }
  }

  return opts;
}

void write_snapshot(const fs::path& directory,
                    int snapshot_id,
                    ConstDoubleView x,
                    ConstDoubleView y,
                    ConstDoubleView h) {
  fs::create_directories(directory);
  std::ostringstream filename;
  filename << "data" << std::setfill('0') << std::setw(3) << snapshot_id << ".csv";
  const auto file = directory / filename.str();

  auto host_x = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), x);
  auto host_y = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), y);
  auto host_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), h);

  std::ofstream output(file);
  if (!output) {
    throw std::runtime_error("Failed to open " + file.string() + " for writing");
  }

  output << std::setprecision(12);
  output << "x,y,h\n";
  const auto n = host_x.extent(0);
  for (size_t i = 0; i < n; ++i) {
    output << host_x(i) << "," << host_y(i) << "," << host_h(i) << "\n";
  }
}

void record_profile(ShallowWaterEquationFused& equation, const Options& opts) {
  const std::string exec_name = ExecSpace::name();

  for (int i = 0; i < opts.profile_warmup; ++i) {
    equation.step();
  }
  Kokkos::fence();

  const auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < opts.profile_iterations; ++i) {
    equation.step();
  }
  Kokkos::fence();
  const auto end = std::chrono::steady_clock::now();

  const double seconds =
      std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count();
  const double ms_per_iter = seconds / opts.profile_iterations * 1000.0;

  std::cout << "Time to run step() " << opts.profile_iterations << " times: " << seconds
            << " seconds, " << ms_per_iter << " ms per iteration.\n";

  fs::path csv_path = opts.write_output ? (opts.output_dir / "timing_fused.csv")
                                        : fs::path("timing_fused.csv");
  if (csv_path.has_parent_path() && !csv_path.parent_path().empty()) {
    fs::create_directories(csv_path.parent_path());
  }

  const bool write_header = !fs::exists(csv_path);
  std::ofstream csv(csv_path, std::ios::app);
  if (!csv) {
    throw std::runtime_error("Failed to open " + csv_path.string() + " for writing");
  }

  std::cout << "Writing profile results to: " << csv_path.string() << std::endl;

  if (write_header) {
    csv << "exec_space,iterations,seconds,ms_per_iteration\n";
  }
  csv << exec_name << "," << opts.profile_iterations << ",";
  csv << std::fixed << std::setprecision(6) << seconds << ",";
  csv << std::setprecision(4) << ms_per_iter << "\n";
  csv.unsetf(std::ios::floatfield);
}

}  // namespace swe

int main(int argc, char* argv[]) {
  Kokkos::ScopeGuard guard(argc, argv);

  swe::Options options;
  try {
    options = swe::parse_args(argc, argv);
  } catch (const std::exception& ex) {
    std::cerr << "Argument error: " << ex.what() << "\n";
    swe::print_usage(argv[0]);
    return 1;
  }

  try {
    auto arrays = swe::load_disk_arrays(options.data_dir);
    swe::ShallowWaterEquationFused equation(arrays, options.dt);

    std::cout << "Loaded mesh with " << equation.cells() << " cells.\n";

    if (options.profile) {
      swe::record_profile(equation, options);
    } else {
      std::cout << "Running " << options.steps << " steps (dt=" << options.dt << ").\n";
      Kokkos::Timer timer;
      for (int step = 0; step < options.steps; ++step) {
        equation.step();

        if (options.write_output && (step % options.output_interval == 0)) {
          swe::write_snapshot(options.output_dir, step / options.output_interval, equation.x(),
                              equation.y(), equation.h());
        }
      }
      Kokkos::fence();
      const double seconds = timer.seconds();
      std::cout << "Simulation finished in " << seconds << " seconds ("
                << (seconds / options.steps) * 1000.0 << " ms/step).\n";
    }

  } catch (const std::exception& ex) {
    std::cerr << "Runtime error: " << ex.what() << "\n";
    return 1;
  }

  return 0;
}

