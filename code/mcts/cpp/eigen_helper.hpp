#pragma once

template <typename DerivedV, typename DerivedB>
auto clip(const Eigen::DenseBase<DerivedV>& v,
          const Eigen::DenseBase<DerivedB>& v_min,
          const Eigen::DenseBase<DerivedB>& v_max)
{
  return v.cwiseMin(v_max).cwiseMax(v_min);
}