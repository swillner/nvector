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

#ifndef NVECTOR_DETAIL_H
#define NVECTOR_DETAIL_H

#include <array>
#include <cstddef>
#include <stdexcept>
#include <tuple>
#include <type_traits>

namespace nvector {

struct Slice;

namespace detail {

template<typename T>
constexpr T multiply_all() {
    return 1;
}
template<typename T, typename... Args>
constexpr T multiply_all(T arg, Args... args) {
    return arg * multiply_all<T>(args...);
}

template<typename Arg>
constexpr bool all_values_equal(Arg&& arg) {
    (void)arg;
    return true;
}
template<typename Arg1, typename Arg2, typename... Args>
constexpr bool all_values_equal(Arg1&& arg1, Arg2&& arg2, Args&&... args) {
    return arg1 == arg2 && all_values_equal(std::forward<Arg2>(arg2), std::forward<Args>(args)...);
}

template<bool...>
struct bool_pack;
template<bool... v>
using all_true = std::is_same<bool_pack<true, v...>, bool_pack<v..., true>>;

template<typename... Args>
using only_for_sizes = typename std::enable_if<all_true<std::is_integral<typename std::remove_reference<Args>::type>::value...>::value>::type;

template<typename... Args>
using only_for_slices = typename std::enable_if<all_true<std::is_same<Slice, typename std::remove_reference<Args>::type>::value...>::value>::type;

template<typename A, typename B>
constexpr B&& map(A&& /* unused */, B&& b) {
    return std::move(b);
}

void pass() {}

template<typename Arg>
void pass(Arg&& /* unused */) {}

template<typename Arg, typename... Args>
void pass(Arg&& /* unused */, Args&&... args) {
    pass(std::forward<Args>(args)...);
}

template<typename Arg>
constexpr bool none_ended(Arg&& it) {
    return !it.ended();
}

template<typename Arg, typename... Args>
constexpr bool none_ended(Arg&& it, Args&&... its) {
    return !it.ended() && none_ended(std::forward<Args>(its)...);
}

template<std::size_t c, std::size_t dim, typename... Args>
struct foreach_dim {
    template<typename... Slices>
    static constexpr bool all_sizes_equal(Slices&&... args) {
        return all_values_equal(std::get<c>(args).size...) && foreach_dim<c + 1, dim>::all_sizes_equal(std::forward<Slices>(args)...);
    }
    static constexpr std::size_t total_size(const std::array<Slice, dim>& dims) { return foreach_dim<c + 1, dim>::total_size(dims) * std::get<c>(dims).size; }
    template<std::size_t... Ns>
    static constexpr std::array<std::size_t, dim> begin() {
        return foreach_dim<c + 1, dim>::template begin<Ns..., 0>();
    }
    template<std::size_t... Ns>
    static constexpr std::array<std::size_t, dim> end(const std::array<Slice, dim>& dims) {
        return foreach_dim<c + 1, dim>::template end<Ns..., c>(dims);
    }
    template<typename Tref, typename Iterator>
    static constexpr Tref dereference(std::size_t index, const std::array<std::size_t, dim>& pos, const std::array<Slice, dim>& dims, Iterator&& it) {
        return foreach_dim<c + 1, dim>::template dereference<Tref>(index + (std::get<c>(pos) + std::get<c>(dims).begin) * std::get<c>(dims).stride, pos, dims,
                                                                   std::forward<Iterator>(it));
    }
    static constexpr void increase(std::array<std::size_t, dim>& pos, const std::array<Slice, dim>& dims) {
        if (std::get<dim - 1 - c>(pos) == std::get<dim - 1 - c>(dims).size - 1) {
            std::get<dim - 1 - c>(pos) = 0;
            foreach_dim<c + 1, dim>::increase(pos, dims);
        } else {
            ++std::get<dim - 1 - c>(pos);
        }
    }
    static constexpr void increase(std::array<std::size_t, dim>& pos, const std::array<Slice, dim>& dims, std::size_t by) {
        const auto d = std::ldiv(std::get<dim - 1 - c>(pos) + by, std::get<dim - 1 - c>(dims).size);
        std::get<dim - 1 - c>(pos) = d.rem;
        if (d.quot > 0) {
            foreach_dim<c + 1, dim>::increase(pos, dims, d.quot);
        }
    }
    template<std::size_t... Ns, typename Function>
    static constexpr void pass_parameters_void(const std::array<std::size_t, dim>& pos, Function&& func, Args&&... args) {
        foreach_dim<c + 1, dim, Args...>::template pass_parameters_void<Ns..., c>(pos, std::forward<Function>(func), std::forward<Args>(args)...);
    }
    template<std::size_t... Ns, typename Function>
    static constexpr bool pass_parameters(const std::array<std::size_t, dim>& pos, Function&& func, Args&&... args) {
        return foreach_dim<c + 1, dim, Args...>::template pass_parameters<Ns..., c>(pos, std::forward<Function>(func), std::forward<Args>(args)...);
    }
};

template<std::size_t dim, typename... Args>
struct foreach_dim<dim, dim, Args...> {
    template<typename... Slices>
    static constexpr bool all_sizes_equal(Slices&&... args) {
        pass(args...);
        return true;
    }
    static constexpr std::size_t total_size(const std::array<Slice, dim>& dims) {
        (void)dims;
        return 1;
    }
    template<std::size_t... Ns>
    static constexpr std::array<std::size_t, dim> begin() {
        return {Ns...};
    }
    template<std::size_t... Ns>
    static constexpr std::array<std::size_t, dim> end(const std::array<Slice, dim>& dims) {
        return {std::get<Ns>(dims).size...};
    }
    template<typename Tref, typename Iterator>
    static constexpr Tref dereference(std::size_t index, const std::array<std::size_t, dim>& pos, const std::array<Slice, dim>& dims, Iterator&& it) {
        (void)pos;
        (void)dims;
        return it[index];
    }
    static constexpr void increase(std::array<std::size_t, dim>& pos, const std::array<Slice, dim>& dims) {
        std::get<0>(pos) = std::get<0>(dims).size;  // reset pos[0] to 1 beyond max value -> represents "end"
    }
    static constexpr void increase(std::array<std::size_t, dim>& pos, const std::array<Slice, dim>& dims, std::size_t by) {
        (void)by;
        std::get<0>(pos) = std::get<0>(dims).size;  // reset pos[0] to 1 beyond max value -> represents "end"
    }
    template<std::size_t... Ns, typename Function>
    static constexpr void pass_parameters_void(const std::array<std::size_t, dim>& pos, Function&& func, Args&&... args) {
        func(std::get<Ns>(pos)..., args...);
    }
    template<std::size_t... Ns, typename Function>
    static constexpr bool pass_parameters(const std::array<std::size_t, dim>& pos, Function&& func, Args&&... args) {
        return func(std::get<Ns>(pos)..., args...);
    }
};

template<typename Function, typename Additional, typename Arg, typename... Args>
constexpr bool loop_foreach_iterator(Function&& func, Additional&& temp, Arg&& it, Args&&... its) {
    (void)temp;  // temp is used to keep views for the iterators in memory for the lifespan of this function call
    while (none_ended(std::forward<Arg>(it), std::forward<Args>(its)...)) {
        if (!foreach_dim<0, Arg::dimensions, typename std::remove_reference<Arg>::type::reference_type,
                         typename std::remove_reference<Args>::type::reference_type...>::template pass_parameters(it.pos(), std::forward<Function>(func), *it,
                                                                                                                  *its...)) {
            return false;
        }
        ++it;
        pass(++its...);
    }
    return true;
}

template<typename Function, typename Arg, typename... Args>
constexpr void loop_foreach_view_parallel(Function&& func, Arg&& view, Args&&... views) {
    if (!foreach_dim<0, view.dimensions>::all_sizes_equal(view.slices(), views.slices()...)) {
        throw std::runtime_error("dimensions of different views have different sizes");
    }
#pragma omp parallel for default(shared)
    for (std::size_t i = 0; i < view.total_size(); ++i) {
        std::array<std::size_t, view.dimensions> pos = foreach_dim<0, view.dimensions>::begin();
        foreach_dim<0, view.dimensions>::increase(pos, view.slices(), i);
        foreach_dim<0, view.dimensions, typename std::remove_reference<Arg>::type::reference_type,
                    typename std::remove_reference<Args>::type::reference_type...>::
            template pass_parameters_void(pos, std::forward<Function>(func),
                                          foreach_dim<0, view.dimensions>::template dereference<typename std::remove_reference<Arg>::type::reference_type>(
                                              0, pos, view.slices(), view.data()),
                                          foreach_dim<0, views.dimensions>::template dereference<typename std::remove_reference<Args>::type::reference_type>(
                                              0, pos, views.slices(), views.data())...);
    }
}

template<typename Function, typename Arg, typename... Args>
constexpr void loop_foreach_aligned_view_parallel(Function&& func, Arg&& view, Args&&... views) {
    if (!all_values_equal(view.slices(), views.slices()...)) {
        throw std::runtime_error("views have different slices");
    }
#pragma omp parallel for default(shared)
    for (std::size_t i = 0; i < view.total_size(); ++i) {
        func(i, view[i], views[i]...);
    }
}

template<std::size_t i, std::size_t n, typename... Args>
struct foreach_helper {
    template<typename Function, std::size_t... Ns>
    static constexpr bool foreach_view(Function&& func, const std::tuple<Args...>& views) {
        return foreach_helper<i + 1, n, Args...>::template foreach_view<Function, Ns..., i>(std::forward<Function>(func), views);
    }
    template<typename Function, std::size_t... Ns>
    static constexpr void foreach_view_parallel(Function&& func, const std::tuple<Args...>& views) {
        foreach_helper<i + 1, n, Args...>::template foreach_view_parallel<Function, Ns..., i>(std::forward<Function>(func), views);
    }
    template<typename Function, std::size_t... Ns>
    static constexpr void foreach_aligned_parallel(Function&& func, const std::tuple<Args...>& views) {
        foreach_helper<i + 1, n, Args...>::template foreach_aligned_parallel<Function, Ns..., i>(std::forward<Function>(func), views);
    }
    template<typename Function, typename Splittype, std::size_t... Ns>
    static constexpr bool foreach_split(Function&& func, const std::tuple<Args...>& views) {
        return foreach_helper<i + 1, n, Args...>::template foreach_split<Function, Splittype, Ns..., i>(std::forward<Function>(func), views);
    }
    template<typename Function, typename Splittype, std::size_t... Ns>
    static constexpr void foreach_split_parallel(Function&& func, const std::tuple<Args...>& views) {
        foreach_helper<i + 1, n, Args...>::template foreach_split_parallel<Function, Splittype, Ns..., i>(std::forward<Function>(func), views);
    }
};

template<typename Function, typename... Args>
constexpr bool foreach_split_helper(Function&& func, Args&&... splits) {
    return loop_foreach_iterator(func, collect(splits...), std::begin(splits)...);
}
template<typename Function, typename... Args>
constexpr void foreach_split_helper_parallel(Function&& func, Args&&... splits) {
    loop_foreach_view_parallel(func, std::forward<Args>(splits)...);
}

template<std::size_t n, typename... Args>
struct foreach_helper<n, n, Args...> {
    template<typename Function, std::size_t... Ns>
    static constexpr bool foreach_view(Function&& func, const std::tuple<Args...>& views) {
        return loop_foreach_iterator(std::forward<Function>(func), 0, std::begin(std::get<Ns>(views))...);
    }
    template<typename Function, std::size_t... Ns>
    static constexpr void foreach_view_parallel(Function&& func, const std::tuple<Args...>& views) {
        loop_foreach_view_parallel(std::forward<Function>(func), std::get<Ns>(views)...);
    }
    template<typename Function, std::size_t... Ns>
    static constexpr void foreach_aligned_parallel(Function&& func, const std::tuple<Args...>& views) {
        loop_foreach_aligned_view_parallel(std::forward<Function>(func), std::get<Ns>(views)...);
    }
    template<typename Function, typename Splittype, std::size_t... Ns>
    static constexpr bool foreach_split(Function&& func, const std::tuple<Args...>& views) {
        return foreach_split_helper(std::forward<Function>(func), std::get<Ns>(views).template split<Splittype>()...);
    }
    template<typename Function, typename Splittype, std::size_t... Ns>
    static constexpr void foreach_split_parallel(Function&& func, const std::tuple<Args...>& views) {
        foreach_split_helper_parallel(std::forward<Function>(func), std::get<Ns>(views).template split<Splittype>()...);
    }
};

}  // namespace detail

}  // namespace nvector

#endif
