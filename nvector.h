/*
  Copyright (C) 2019 Sven Willner <sven.willner@gmail.com>

  This program is free software: you can redistribute it and/or modify
  it under the terms of the GNU Affero General Public License as published
  by the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  This program is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU Affero General Public License for more details.

  You should have received a copy of the GNU Affero General Public License
  along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef NVECTOR_H
#define NVECTOR_H

#include <array>
#include <cstddef>
#include <tuple>
#include <type_traits>
#include <utility>
#include "Vector.h"
#include "View.h"
#include "detail.h"

namespace nvector {

template<typename... Args>
constexpr std::tuple<const Args&...> collect(const Args&... args) {
    return std::tuple<const Args&...>(args...);
}

template<typename... Args>
constexpr std::tuple<Args&...> collect(Args&... args) {
    return std::tuple<Args&...>(args...);
}

struct Slice {
    long begin = 0;
    std::size_t size = 0;
    long stride = 1;
    constexpr Slice(long begin_p, std::size_t size_p, long stride_p) : begin(begin_p), size(size_p), stride(stride_p) {}
    constexpr Slice() = default;
    constexpr bool operator==(const Slice& other) const { return begin == other.begin && size == other.size && stride == other.stride; }
};

template<typename... Args, typename Function>
constexpr bool foreach_view(const std::tuple<Args...>& views, Function&& func) {
    return detail::foreach_helper<0, std::tuple_size<std::tuple<Args...>>::value, Args...>::foreach_view(std::forward<Function>(func), views);
}

template<typename... Args, typename Function>
constexpr void foreach_view_parallel(const std::tuple<Args...>& views, Function&& func) {
    detail::foreach_helper<0, std::tuple_size<std::tuple<Args...>>::value, Args...>::foreach_view_parallel(std::forward<Function>(func), views);
}

template<typename... Args, typename Function>
constexpr void foreach_aligned_parallel(const std::tuple<Args...>& views, Function&& func) {
    detail::foreach_helper<0, std::tuple_size<std::tuple<Args...>>::value, Args...>::foreach_aligned_parallel(std::forward<Function>(func), views);
}

template<typename Splittype, typename... Args, typename Function>
constexpr bool foreach_split(const std::tuple<Args...>& views, Function&& func) {
    return detail::foreach_helper<0, sizeof...(Args), Args...>::template foreach_split<Function, Splittype>(std::forward<Function>(func), views);
}

template<typename Splittype, typename... Args, typename Function>
constexpr void foreach_split_parallel(const std::tuple<Args...>& views, Function&& func) {
    detail::foreach_helper<0, sizeof...(Args), Args...>::template foreach_split_parallel<Function, Splittype>(std::forward<Function>(func), views);
}

}  // namespace nvector

#endif
