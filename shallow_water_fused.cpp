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
#include <thread>

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

class ShallowWaterEquationFused {
 public:
  explicit ShallowWaterEquationFused(const DiskArrays& arrays, double dt);
  void step();
  int cells() const { return nc_; }
  ConstDoubleView h() const { return h_; }
  ConstDoubleView x() const { return x_; }
  ConstDoubleView y() const { return y_; }

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

  // torch.fx / EASIER graph intermediates.
  DoubleView truediv_2_;
  DoubleView truediv_3_;
  DoubleView truediv_4_;
  DoubleView truediv_7_;
  DoubleView truediv_8_;
  DoubleView truediv_9_;
  DoubleView truediv_12_;
  DoubleView truediv_13_;
  DoubleView truediv_14_;

  DoubleView add_10_;
  DoubleView add_11_;
  DoubleView add_12_;
  DoubleView add_23_;
  DoubleView add_24_;
  DoubleView add_25_;
  DoubleView add_36_;
  DoubleView add_37_;
  DoubleView add_38_;

  DoubleView scatter_;
  DoubleView scatter_1_;
  DoubleView scatter_2_;
  DoubleView scatter_3_;
  DoubleView scatter_4_;
  DoubleView scatter_5_;
  DoubleView scatter_6_;
  DoubleView scatter_7_;
  DoubleView scatter_8_;
  DoubleView scatter_9_;
  DoubleView scatter_10_;
  DoubleView scatter_11_;

  DoubleView scatter_b_;
  DoubleView scatter_b_1_;
  DoubleView scatter_b_2_;
  DoubleView scatter_b_3_;
  DoubleView scatter_b_4_;
  DoubleView scatter_b_5_;
  DoubleView scatter_b_6_;
  DoubleView scatter_b_7_;
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
      y_("y", nc_) {
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

  truediv_2_ = DoubleView("truediv_2", nc_);
  truediv_3_ = DoubleView("truediv_3", nc_);
  truediv_4_ = DoubleView("truediv_4", nc_);
  truediv_7_ = DoubleView("truediv_7", nc_);
  truediv_8_ = DoubleView("truediv_8", nc_);
  truediv_9_ = DoubleView("truediv_9", nc_);
  truediv_12_ = DoubleView("truediv_12", nc_);
  truediv_13_ = DoubleView("truediv_13", nc_);
  truediv_14_ = DoubleView("truediv_14", nc_);

  add_10_ = DoubleView("add_10", nc_);
  add_11_ = DoubleView("add_11", nc_);
  add_12_ = DoubleView("add_12", nc_);
  add_23_ = DoubleView("add_23", nc_);
  add_24_ = DoubleView("add_24", nc_);
  add_25_ = DoubleView("add_25", nc_);
  add_36_ = DoubleView("add_36", nc_);
  add_37_ = DoubleView("add_37", nc_);
  add_38_ = DoubleView("add_38", nc_);

  scatter_ = DoubleView("scatter", nc_);
  scatter_1_ = DoubleView("scatter_1", nc_);
  scatter_2_ = DoubleView("scatter_2", nc_);
  scatter_3_ = DoubleView("scatter_3", nc_);
  scatter_4_ = DoubleView("scatter_4", nc_);
  scatter_5_ = DoubleView("scatter_5", nc_);
  scatter_6_ = DoubleView("scatter_6", nc_);
  scatter_7_ = DoubleView("scatter_7", nc_);
  scatter_8_ = DoubleView("scatter_8", nc_);
  scatter_9_ = DoubleView("scatter_9", nc_);
  scatter_10_ = DoubleView("scatter_10", nc_);
  scatter_11_ = DoubleView("scatter_11", nc_);

  scatter_b_ = DoubleView("scatter_b", nc_);
  scatter_b_1_ = DoubleView("scatter_b_1", nc_);
  scatter_b_2_ = DoubleView("scatter_b_2", nc_);
  scatter_b_3_ = DoubleView("scatter_b_3", nc_);
  scatter_b_4_ = DoubleView("scatter_b_4", nc_);
  scatter_b_5_ = DoubleView("scatter_b_5", nc_);
  scatter_b_6_ = DoubleView("scatter_b_6", nc_);
  scatter_b_7_ = DoubleView("scatter_b_7", nc_);
}

void ShallowWaterEquationFused::step() {
  const double half_dt = 0.5 * dt_;
  const double full_dt = dt_;
  const double rk_weight = dt_ / 6.0;

  auto src = src_;
  auto dst = dst_;
  auto bcells = bcells_;
  auto alpha = alpha_;
  auto area = area_;
  auto sx = sx_;
  auto sy = sy_;
  auto bsx = bsx_;
  auto bsy = bsy_;

  auto h = h_;
  auto uh = uh_;
  auto vh = vh_;

  auto truediv_2 = truediv_2_;
  auto truediv_3 = truediv_3_;
  auto truediv_4 = truediv_4_;
  auto truediv_7 = truediv_7_;
  auto truediv_8 = truediv_8_;
  auto truediv_9 = truediv_9_;
  auto truediv_12 = truediv_12_;
  auto truediv_13 = truediv_13_;
  auto truediv_14 = truediv_14_;

  auto add_10 = add_10_;
  auto add_11 = add_11_;
  auto add_12 = add_12_;
  auto add_23 = add_23_;
  auto add_24 = add_24_;
  auto add_25 = add_25_;
  auto add_36 = add_36_;
  auto add_37 = add_37_;
  auto add_38 = add_38_;

  auto scatter = scatter_;
  auto scatter_1 = scatter_1_;
  auto scatter_2 = scatter_2_;
  auto scatter_3 = scatter_3_;
  auto scatter_4 = scatter_4_;
  auto scatter_5 = scatter_5_;
  auto scatter_6 = scatter_6_;
  auto scatter_7 = scatter_7_;
  auto scatter_8 = scatter_8_;
  auto scatter_9 = scatter_9_;
  auto scatter_10 = scatter_10_;
  auto scatter_11 = scatter_11_;

  auto scatter_b = scatter_b_;
  auto scatter_b_1 = scatter_b_1_;
  auto scatter_b_2 = scatter_b_2_;
  auto scatter_b_3 = scatter_b_3_;
  auto scatter_b_4 = scatter_b_4_;
  auto scatter_b_5 = scatter_b_5_;
  auto scatter_b_6 = scatter_b_6_;
  auto scatter_b_7 = scatter_b_7_;

  Kokkos::deep_copy(scatter_b, 0.0);
  Kokkos::deep_copy(scatter_b_1, 0.0);
  // torch.fx: easier0_select_reduce30
  Kokkos::parallel_for("easier0_select_reduce30", RangePolicy(0, nbc_),
                       KOKKOS_LAMBDA(const int i) {
                         const int cell = static_cast<int>(bcells(i));
                         const double h_cell = h(cell);
                         const double p = 0.5 * h_cell * h_cell;
                         Kokkos::atomic_add(&scatter_b(cell), p * bsx(i));
                         Kokkos::atomic_add(&scatter_b_1(cell), p * bsy(i));
                       });

  Kokkos::deep_copy(scatter, 0.0);
  Kokkos::deep_copy(scatter_1, 0.0);
  Kokkos::deep_copy(scatter_2, 0.0);
  // torch.fx: easier1_select_reduce16
  Kokkos::parallel_for("easier1_select_reduce16", RangePolicy(0, ne_),
                       KOKKOS_LAMBDA(const int e) {
                         const int left = static_cast<int>(src(e));
                         const int right = static_cast<int>(dst(e));
                         const double a = alpha(e);

                         const double h_face = (1.0 - a) * h(left) + a * h(right);
                         const double uh_face = (1.0 - a) * uh(left) + a * uh(right);
                         const double vh_face = (1.0 - a) * vh(left) + a * vh(right);

                         const double u = uh_face / h_face;
                         const double v = vh_face / h_face;
                         const double p = 0.5 * h_face * h_face;
                         const double sx_val = sx(e);
                         const double sy_val = sy(e);

                         const double mass_flux = uh_face * sx_val + vh_face * sy_val;
                         const double x_momentum_flux =
                             (u * uh_face + p) * sx_val + u * vh_face * sy_val;
                         const double y_momentum_flux =
                             v * uh_face * sx_val + (v * vh_face + p) * sy_val;

                         Kokkos::atomic_add(&scatter(right), mass_flux);
                         Kokkos::atomic_add(&scatter_1(right), x_momentum_flux);
                         Kokkos::atomic_add(&scatter_2(right), y_momentum_flux);
                       });

  // torch.fx: easier2_map35
  Kokkos::parallel_for("easier2_map35", RangePolicy(0, nc_), KOKKOS_LAMBDA(const int i) {
    truediv_2(i) = -scatter(i) / area(i);
    add_10(i) = h(i) + half_dt * truediv_2(i);
  });

  // torch.fx: easier3_map47
  Kokkos::parallel_for("easier3_map47", RangePolicy(0, nc_), KOKKOS_LAMBDA(const int i) {
    truediv_3(i) = -(scatter_1(i) + scatter_b(i)) / area(i);
    add_11(i) = uh(i) + half_dt * truediv_3(i);
  });

  // torch.fx: easier4_map59
  Kokkos::parallel_for("easier4_map59", RangePolicy(0, nc_), KOKKOS_LAMBDA(const int i) {
    truediv_4(i) = -(scatter_2(i) + scatter_b_1(i)) / area(i);
    add_12(i) = vh(i) + half_dt * truediv_4(i);
  });

  Kokkos::deep_copy(scatter_b_2, 0.0);
  Kokkos::deep_copy(scatter_b_3, 0.0);
  // torch.fx: easier5_select_reduce92
  Kokkos::parallel_for("easier5_select_reduce92", RangePolicy(0, nbc_),
                       KOKKOS_LAMBDA(const int i) {
                         const int cell = static_cast<int>(bcells(i));
                         const double h_cell = add_10(cell);
                         const double p = 0.5 * h_cell * h_cell;
                         Kokkos::atomic_add(&scatter_b_2(cell), p * bsx(i));
                         Kokkos::atomic_add(&scatter_b_3(cell), p * bsy(i));
                       });

  Kokkos::deep_copy(scatter_4, 0.0);
  Kokkos::deep_copy(scatter_3, 0.0);
  Kokkos::deep_copy(scatter_5, 0.0);
  // torch.fx: easier6_select_reduce80
  // Preserve output order exactly: [scatter_4, scatter_3, scatter_5].
  Kokkos::parallel_for("easier6_select_reduce80", RangePolicy(0, ne_),
                       KOKKOS_LAMBDA(const int e) {
                         const int left = static_cast<int>(src(e));
                         const int right = static_cast<int>(dst(e));
                         const double a = alpha(e);

                         const double h_face = (1.0 - a) * add_10(left) + a * add_10(right);
                         const double uh_face = (1.0 - a) * add_11(left) + a * add_11(right);
                         const double vh_face = (1.0 - a) * add_12(left) + a * add_12(right);

                         const double u = uh_face / h_face;
                         const double v = vh_face / h_face;
                         const double p = 0.5 * h_face * h_face;
                         const double sx_val = sx(e);
                         const double sy_val = sy(e);

                         const double mass_flux = uh_face * sx_val + vh_face * sy_val;
                         const double x_momentum_flux =
                             (u * uh_face + p) * sx_val + u * vh_face * sy_val;
                         const double y_momentum_flux =
                             v * uh_face * sx_val + (v * vh_face + p) * sy_val;

                         Kokkos::atomic_add(&scatter_4(right), x_momentum_flux);
                         Kokkos::atomic_add(&scatter_3(right), mass_flux);
                         Kokkos::atomic_add(&scatter_5(right), y_momentum_flux);
                       });

  // torch.fx: easier7_map97
  Kokkos::parallel_for("easier7_map97", RangePolicy(0, nc_), KOKKOS_LAMBDA(const int i) {
    truediv_7(i) = -scatter_3(i) / area(i);
    add_23(i) = h(i) + half_dt * truediv_7(i);
  });

  // torch.fx: easier8_map118
  Kokkos::parallel_for("easier8_map118", RangePolicy(0, nc_), KOKKOS_LAMBDA(const int i) {
    truediv_9(i) = -(scatter_5(i) + scatter_b_3(i)) / area(i);
    add_25(i) = vh(i) + half_dt * truediv_9(i);
  });

  // torch.fx: easier9_map107
  Kokkos::parallel_for("easier9_map107", RangePolicy(0, nc_), KOKKOS_LAMBDA(const int i) {
    truediv_8(i) = -(scatter_4(i) + scatter_b_2(i)) / area(i);
    add_24(i) = uh(i) + half_dt * truediv_8(i);
  });

  Kokkos::deep_copy(scatter_b_4, 0.0);
  Kokkos::deep_copy(scatter_b_5, 0.0);
  // torch.fx: easier10_select_reduce151
  Kokkos::parallel_for("easier10_select_reduce151", RangePolicy(0, nbc_),
                       KOKKOS_LAMBDA(const int i) {
                         const int cell = static_cast<int>(bcells(i));
                         const double h_cell = add_23(cell);
                         const double p = 0.5 * h_cell * h_cell;
                         Kokkos::atomic_add(&scatter_b_4(cell), p * bsx(i));
                         Kokkos::atomic_add(&scatter_b_5(cell), p * bsy(i));
                       });

  Kokkos::deep_copy(scatter_7, 0.0);
  Kokkos::deep_copy(scatter_6, 0.0);
  Kokkos::deep_copy(scatter_8, 0.0);
  // torch.fx: easier11_select_reduce139
  // Preserve output order exactly: [scatter_7, scatter_6, scatter_8].
  Kokkos::parallel_for("easier11_select_reduce139", RangePolicy(0, ne_),
                       KOKKOS_LAMBDA(const int e) {
                         const int left = static_cast<int>(src(e));
                         const int right = static_cast<int>(dst(e));
                         const double a = alpha(e);

                         const double h_face = (1.0 - a) * add_23(left) + a * add_23(right);
                         const double uh_face = (1.0 - a) * add_24(left) + a * add_24(right);
                         const double vh_face = (1.0 - a) * add_25(left) + a * add_25(right);

                         const double u = uh_face / h_face;
                         const double v = vh_face / h_face;
                         const double p = 0.5 * h_face * h_face;
                         const double sx_val = sx(e);
                         const double sy_val = sy(e);

                         const double mass_flux = uh_face * sx_val + vh_face * sy_val;
                         const double x_momentum_flux =
                             (u * uh_face + p) * sx_val + u * vh_face * sy_val;
                         const double y_momentum_flux =
                             v * uh_face * sx_val + (v * vh_face + p) * sy_val;

                         Kokkos::atomic_add(&scatter_7(right), x_momentum_flux);
                         Kokkos::atomic_add(&scatter_6(right), mass_flux);
                         Kokkos::atomic_add(&scatter_8(right), y_momentum_flux);
                       });

  // torch.fx: easier12_map177
  Kokkos::parallel_for("easier12_map177", RangePolicy(0, nc_), KOKKOS_LAMBDA(const int i) {
    truediv_14(i) = -(scatter_8(i) + scatter_b_5(i)) / area(i);
    add_38(i) = vh(i) + full_dt * truediv_14(i);
  });

  // torch.fx: easier13_map166
  Kokkos::parallel_for("easier13_map166", RangePolicy(0, nc_), KOKKOS_LAMBDA(const int i) {
    truediv_13(i) = -(scatter_7(i) + scatter_b_4(i)) / area(i);
    add_37(i) = uh(i) + full_dt * truediv_13(i);
  });

  // torch.fx: easier14_map156
  Kokkos::parallel_for("easier14_map156", RangePolicy(0, nc_), KOKKOS_LAMBDA(const int i) {
    truediv_12(i) = -scatter_6(i) / area(i);
    add_36(i) = h(i) + full_dt * truediv_12(i);
  });

  Kokkos::deep_copy(scatter_10, 0.0);
  Kokkos::deep_copy(scatter_9, 0.0);
  Kokkos::deep_copy(scatter_11, 0.0);
  // torch.fx: easier15_select_reduce198
  // Preserve output order exactly: [scatter_10, scatter_9, scatter_11].
  Kokkos::parallel_for("easier15_select_reduce198", RangePolicy(0, ne_),
                       KOKKOS_LAMBDA(const int e) {
                         const int left = static_cast<int>(src(e));
                         const int right = static_cast<int>(dst(e));
                         const double a = alpha(e);

                         const double h_face = (1.0 - a) * add_36(left) + a * add_36(right);
                         const double uh_face = (1.0 - a) * add_37(left) + a * add_37(right);
                         const double vh_face = (1.0 - a) * add_38(left) + a * add_38(right);

                         const double u = uh_face / h_face;
                         const double v = vh_face / h_face;
                         const double p = 0.5 * h_face * h_face;
                         const double sx_val = sx(e);
                         const double sy_val = sy(e);

                         const double mass_flux = uh_face * sx_val + vh_face * sy_val;
                         const double x_momentum_flux =
                             (u * uh_face + p) * sx_val + u * vh_face * sy_val;
                         const double y_momentum_flux =
                             v * uh_face * sx_val + (v * vh_face + p) * sy_val;

                         Kokkos::atomic_add(&scatter_10(right), x_momentum_flux);
                         Kokkos::atomic_add(&scatter_9(right), mass_flux);
                         Kokkos::atomic_add(&scatter_11(right), y_momentum_flux);
                       });

  Kokkos::deep_copy(scatter_b_6, 0.0);
  Kokkos::deep_copy(scatter_b_7, 0.0);
  // torch.fx: easier16_select_reduce210
  Kokkos::parallel_for("easier16_select_reduce210", RangePolicy(0, nbc_),
                       KOKKOS_LAMBDA(const int i) {
                         const int cell = static_cast<int>(bcells(i));
                         const double h_cell = add_36(cell);
                         const double p = 0.5 * h_cell * h_cell;
                         Kokkos::atomic_add(&scatter_b_6(cell), p * bsx(i));
                         Kokkos::atomic_add(&scatter_b_7(cell), p * bsy(i));
                       });

  // torch.fx: easier17_map239
  Kokkos::parallel_for("easier17_map239", RangePolicy(0, nc_), KOKKOS_LAMBDA(const int i) {
    const double truediv_17 = -scatter_9(i) / area(i);
    h(i) += rk_weight * (truediv_2(i) + truediv_7(i) + truediv_12(i) + truediv_17);
  });

  // torch.fx: easier18_map249
  Kokkos::parallel_for("easier18_map249", RangePolicy(0, nc_), KOKKOS_LAMBDA(const int i) {
    const double truediv_19 = -(scatter_11(i) + scatter_b_7(i)) / area(i);
    vh(i) += rk_weight * (truediv_4(i) + truediv_9(i) + truediv_14(i) + truediv_19);
  });

  // torch.fx: easier19_map244
  Kokkos::parallel_for("easier19_map244", RangePolicy(0, nc_), KOKKOS_LAMBDA(const int i) {
    const double truediv_18 = -(scatter_10(i) + scatter_b_6(i)) / area(i);
    uh(i) += rk_weight * (truediv_3(i) + truediv_8(i) + truediv_13(i) + truediv_18);
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
      std::this_thread::sleep_for(std::chrono::seconds(5));
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
