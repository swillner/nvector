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

#ifndef NVECTOR_VECTOR_H
#define NVECTOR_VECTOR_H

#include <algorithm>
#include <array>
#include <cstddef>
#include <iterator>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>
#include "View.h"
#include "detail.h"

namespace nvector {

template<typename T, std::size_t dim, class Storage = std::vector<T>, typename Iterator = typename Storage::iterator>
class Vector : public View<T, dim, Iterator> {
  protected:
    using View<T, dim, Iterator>::it;
    Storage data_m;

  public:
    using View<T, dim, Iterator>::dimensions;
    using View<T, dim, Iterator>::reference_type;
    using View<T, dim, Iterator>::type;
    using View<T, dim, Iterator>::iterator_type;
    using View<T, dim, Iterator>::split_type;

    constexpr Vector() { it = std::begin(data_m); }
    constexpr Vector(const T& initial_value, std::array<Slice, dim> dims_p) : View<T, dim, Iterator>(std::begin(data_m), std::move(dims_p)) {
        data_m.resize(this->total_size(), initial_value);
        it = std::begin(data_m);
    }

    template<typename... Args, typename detail::only_for_sizes<Args...>* = nullptr>
    explicit Vector(const T& initial_value, Args&&... args) {
        this->template initialize_sizes<0>(std::forward<Args>(args)...);
        data_m.resize(detail::multiply_all(std::forward<Args>(args)...), initial_value);
        it = std::begin(data_m);
    }

    template<typename... Args, typename detail::only_for_slices<Args...>* = nullptr>
    explicit Vector(const T& initial_value, Args&&... args) {
        this->template initialize_slices<0>(std::forward<Args>(args)...);
        data_m.resize(detail::multiply_all(args.size...), initial_value);
        it = std::begin(data_m);
    }

    template<typename... Args, typename detail::only_for_sizes<Args...>* = nullptr>
    explicit Vector(Storage data_p, Args&&... args) : data_m(std::move(data_p)) {
        this->template initialize_sizes<0>(std::forward<Args>(args)...);
        if (detail::multiply_all<std::size_t>(std::forward<Args>(args)...) != data_m.size()) {
            throw std::runtime_error("wrong size of underlying data");
        }
        it = std::begin(data_m);
    }

    template<typename... Args, typename detail::only_for_slices<Args...>* = nullptr>
    explicit Vector(Storage data_p, Args&&... args) : data_m(std::move(data_p)) {
        this->template initialize_slices<0>(std::forward<Args>(args)...);
        if (detail::multiply_all<std::size_t>(args.size...) != data_m.size()) {
            throw std::runtime_error("wrong size of underlying data");
        }
        it = std::begin(data_m);
    }

    template<typename... Args>
    constexpr void resize(const T& initial_value, Args&&... args) {
        this->template initialize_sizes<0>(std::forward<Args>(args)...);
        data_m.resize(detail::multiply_all(std::forward<Args>(args)...), initial_value);
        it = std::begin(data_m);
    }

    constexpr void reset(const T& initial_value) { std::fill(std::begin(data_m), std::end(data_m), initial_value); }

    constexpr Storage& data() { return data_m; }
    constexpr const Storage& data() const { return data_m; }
};

}  // namespace nvector

#endif
