#pragma once

template <typename DerivedV, typename DerivedB>
auto clip(const Eigen::ArrayBase<DerivedV>& v,
          const Eigen::ArrayBase<DerivedB>& v_min,
          const Eigen::ArrayBase<DerivedB>& v_max)
{
  return v.min(v_max).max(v_min);
}