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

#ifndef NVECTOR_VIEW_H
#define NVECTOR_VIEW_H

#include <algorithm>
#include <array>
#include <cstddef>
#include <iterator>
#include <stdexcept>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>
#include "detail.h"

namespace nvector {

class Slice;

template<bool... Args>
struct Split {
    static const std::size_t inner_dim = 0;
    static const std::size_t outer_dim = 0;
};

template<bool... Args>
struct Split<true, Args...> {
    static const std::size_t inner_dim = 1 + Split<Args...>::inner_dim;
    static const std::size_t outer_dim = Split<Args...>::outer_dim;
};

template<bool... Args>
struct Split<false, Args...> {
    static const std::size_t inner_dim = Split<Args...>::inner_dim;
    static const std::size_t outer_dim = 1 + Split<Args...>::outer_dim;
};

template<typename T, std::size_t dim, class Iterator = typename std::vector<T>::iterator, typename Tref = typename std::add_lvalue_reference<T>::type>
class View {
  public:
    template<std::size_t inner_dim>
    class SplitViewHandler {
      protected:
        Iterator it;
        std::array<Slice, inner_dim> dims;

      public:
        constexpr SplitViewHandler(Iterator it_p, std::array<Slice, inner_dim> dims_p) : it(std::move(it_p)), dims(std::move(dims_p)){};
        View<T, inner_dim, Iterator, Tref> operator[](std::size_t index) { return {it + index, dims}; }
        constexpr const View<T, inner_dim, Iterator, Tref> operator[](std::size_t index) const { return {it + index, dims}; }
    };

    template<std::size_t inner_dim, std::size_t outer_dim>
    using SplitView = View<View<T, inner_dim, Iterator, Tref>, outer_dim, SplitViewHandler<inner_dim>, View<T, inner_dim, Iterator, Tref>>;

    class iterator : public std::iterator<std::output_iterator_tag, T> {
        friend class View;

      protected:
        View* view = nullptr;
        std::array<std::size_t, dim> pos_m;
        std::size_t total_index = 0;
        const std::size_t end_index = 0;
        constexpr iterator(View* view_p, std::array<std::size_t, dim> pos_p, std::size_t total_index_p, std::size_t end_index_p)
            : view(view_p), pos_m(pos_p), total_index(total_index_p), end_index(end_index_p){};

      public:
        static constexpr std::size_t dimensions = dim;
        using type = T;
        using reference_type = Tref;

        static constexpr iterator begin(View* view_p) { return {view_p, detail::foreach_dim<0, dim>::begin(), 0, view_p->total_size()}; }
        static constexpr iterator end(View* view_p) {
            return {view_p, detail::foreach_dim<0, dim>::end(view_p->dims), view_p->total_size(), view_p->total_size()};
        }
        constexpr bool ended() const { return total_index == end_index; }
        constexpr std::size_t get_end_index() const { return end_index; }
        constexpr std::size_t get_index() const { return total_index; }
        constexpr const std::array<std::size_t, dim>& pos() const { return pos_m; }
        constexpr Tref operator*() const { return detail::foreach_dim<0, dim>::template dereference<Tref>(0, pos_m, view->dims, view->it); }
        iterator operator++() {
            detail::foreach_dim<0, dim>::increase(pos_m, view->dims);
            ++total_index;
            return *this;
        }
        constexpr iterator operator+(std::size_t i) const {
            if (total_index + i >= end_index) {
                return end(view);
            }
            std::array<std::size_t, dim> pos_l = pos_m;
            detail::foreach_dim<0, dim>::increase(pos_l, view->dims, i);
            return iterator(view, std::move(pos_l), total_index + i, end_index);
        }
        constexpr bool operator<(const iterator& other) const { return total_index < other.total_index; }
        constexpr bool operator<(std::size_t other) const { return total_index < other; }
        constexpr bool operator==(const iterator& other) const { return total_index == other.total_index; }
        constexpr bool operator!=(const iterator& other) const { return total_index != other.total_index; }
    };
    friend class iterator;

    class const_iterator : public std::iterator<std::output_iterator_tag, T> {
        friend class View;

      protected:
        const View* const view = nullptr;
        std::array<std::size_t, dim> pos_m;
        std::size_t total_index = 0;
        const std::size_t end_index = 0;
        constexpr const_iterator(const View* const view_p, std::array<std::size_t, dim> pos_p, std::size_t total_index_p, std::size_t end_index_p)
            : view(view_p), pos_m(pos_p), total_index(total_index_p), end_index(end_index_p){};

      public:
        static constexpr std::size_t dimensions = dim;
        using type = T;
        using reference_type = Tref;

        static constexpr const_iterator begin(const View* const view_p) { return {view_p, detail::foreach_dim<0, dim>::begin(), 0, view_p->total_size()}; }
        static constexpr const_iterator end(const View* const view_p) {
            return {view_p, detail::foreach_dim<0, dim>::end(view_p->dims), view_p->total_size(), view_p->total_size()};
        }
        constexpr bool ended() const { return total_index == end_index; }
        constexpr std::size_t get_end_index() const { return end_index; }
        constexpr std::size_t get_index() const { return total_index; }
        constexpr const std::array<std::size_t, dim>& pos() const { return pos_m; }
        constexpr const Tref operator*() const { return detail::foreach_dim<0, dim>::template dereference<Tref>(0, pos_m, view->dims, view->it); }
        const_iterator operator++() {
            detail::foreach_dim<0, dim>::increase(pos_m, view->dims);
            ++total_index;
            return *this;
        }
        constexpr const_iterator operator+(std::size_t i) const {
            if (total_index + i >= end_index) {
                return end(view);
            }
            std::array<std::size_t, dim> pos_l = pos_m;
            detail::foreach_dim<0, dim>::increase(pos_l, view->dims, i);
            return const_iterator(view, std::move(pos_l), total_index + i, end_index);
        }
        constexpr bool operator<(const const_iterator& other) const { return total_index < other.total_index; }
        constexpr bool operator<(std::size_t other) const { return total_index < other; }
        constexpr bool operator==(const const_iterator& other) const { return total_index == other.total_index; }
        constexpr bool operator!=(const const_iterator& other) const { return total_index != other.total_index; }
    };
    friend class const_iterator;

  protected:
    std::array<Slice, dim> dims;
    Iterator it;

    template<std::size_t c, typename... Args>
    Tref i_(std::size_t index, const std::size_t& i, Args&&... args) noexcept {
        return i_<c + 1>(index + (i + std::get<c>(dims).begin) * std::get<c>(dims).stride, std::forward<Args>(args)...);
    }

    template<std::size_t c>
    Tref i_(std::size_t index) noexcept {
        static_assert(c == dim, "wrong number of arguments");
        return it[index];
    }

    template<std::size_t c, typename... Args>
    constexpr const Tref i_(std::size_t index, const std::size_t& i, Args&&... args) const noexcept {
        return i_<c + 1>(index + (i + std::get<c>(dims).begin) * std::get<c>(dims).stride, std::forward<Args>(args)...);
    }

    template<std::size_t c>
    constexpr const Tref i_(std::size_t index) const noexcept {
        static_assert(c == dim, "wrong number of arguments");
        return it[index];
    }

    template<std::size_t c, typename... Args>
    Tref at_(std::size_t index, const std::size_t& i, Args&&... args) {
        if (i >= std::get<c>(dims).size) {
            throw std::out_of_range("index out of bounds");
        }
        return at_<c + 1>(index + (i + std::get<c>(dims).begin) * std::get<c>(dims).stride, std::forward<Args>(args)...);
    }

    template<std::size_t c>
    Tref at_(std::size_t index) {
        static_assert(c == dim, "wrong number of arguments");
        return it[index];
    }

    template<std::size_t c, typename... Args>
    constexpr const Tref at_(std::size_t index, const std::size_t& i, Args&&... args) const {
        if (i >= std::get<c>(dims).size) {
            throw std::out_of_range("index out of bounds");
        }
        return at_<c + 1>(index + (i + std::get<c>(dims).begin) * std::get<c>(dims).stride, std::forward<Args>(args)...);
    }

    template<std::size_t c>
    constexpr const Tref at_(std::size_t index) const {
        static_assert(c == dim, "wrong number of arguments");
        return it[index];
    }

    template<std::size_t c, typename... Args>
    void initialize_slices(const Slice& i, Args&&... args) {
        std::get<c>(dims) = i;
        initialize_slices<c + 1>(std::forward<Args>(args)...);
    }

    template<std::size_t c>
    void initialize_slices() {
        static_assert(c == dim, "wrong number of arguments");
    }

    template<std::size_t c, typename... Args>
    void initialize_sizes(std::size_t size, Args&&... args) {
        std::get<c>(dims) = {0, size, detail::multiply_all<int>(std::forward<Args>(args)...)};
        initialize_sizes<c + 1>(std::forward<Args>(args)...);
    }

    template<std::size_t c>
    void initialize_sizes() {
        static_assert(c == dim, "wrong number of arguments");
    }

    template<std::size_t c, std::size_t inner_c, std::size_t inner_dim, std::size_t outer_c, std::size_t outer_dim, bool... Args>
    struct splitter {
        static constexpr SplitView<inner_dim, outer_dim> split(Iterator it,
                                                               std::array<Slice, inner_dim> inner_dims,
                                                               std::array<Slice, outer_dim> outer_dims,
                                                               const std::array<Slice, dim>& dims);
    };

    template<std::size_t c, std::size_t inner_c, std::size_t inner_dim, std::size_t outer_c, std::size_t outer_dim, bool... Args>
    struct splitter<c, inner_c, inner_dim, outer_c, outer_dim, true, Args...> {
        static constexpr SplitView<inner_dim, outer_dim> split(Iterator it,
                                                               std::array<Slice, inner_dim> inner_dims,
                                                               std::array<Slice, outer_dim> outer_dims,
                                                               const std::array<Slice, dim>& dims) {
            std::get<inner_c>(inner_dims) = std::get<c>(dims);
            return splitter<c + 1, inner_c + 1, inner_dim, outer_c, outer_dim, Args...>::split(std::move(it), std::move(inner_dims), std::move(outer_dims),
                                                                                               dims);
        }
    };

    template<std::size_t c, std::size_t inner_c, std::size_t inner_dim, std::size_t outer_c, std::size_t outer_dim, bool... Args>
    struct splitter<c, inner_c, inner_dim, outer_c, outer_dim, false, Args...> {
        static constexpr SplitView<inner_dim, outer_dim> split(Iterator it,
                                                               std::array<Slice, inner_dim> inner_dims,
                                                               std::array<Slice, outer_dim> outer_dims,
                                                               const std::array<Slice, dim>& dims) {
            std::get<outer_c>(outer_dims) = std::get<c>(dims);
            return splitter<c + 1, inner_c, inner_dim, outer_c + 1, outer_dim, Args...>::split(std::move(it), std::move(inner_dims), std::move(outer_dims),
                                                                                               dims);
        }
    };

    template<std::size_t c, std::size_t inner_c, std::size_t inner_dim, std::size_t outer_c, std::size_t outer_dim>
    struct splitter<c, inner_c, inner_dim, outer_c, outer_dim> {
        static constexpr SplitView<inner_dim, outer_dim> split(Iterator it,
                                                               std::array<Slice, inner_dim> inner_dims,
                                                               std::array<Slice, outer_dim> outer_dims,
                                                               const std::array<Slice, dim>& dims) {
            (void)dims;
            return SplitView<inner_dim, outer_dim>(std::move(SplitViewHandler<inner_dim>(std::move(it), std::move(inner_dims))), std::move(outer_dims));
        }
    };

    template<typename Splittype>
    struct splitter_helper {};

    template<bool... Args>
    struct splitter_helper<Split<Args...>> {
        static constexpr SplitView<Split<Args...>::inner_dim, Split<Args...>::outer_dim> split(Iterator it, const std::array<Slice, dim>& dims) {
            const std::size_t inner_dim = Split<Args...>::inner_dim;
            const std::size_t outer_dim = Split<Args...>::outer_dim;
            static_assert(inner_dim + outer_dim == dim, "wrong number of arguments");
            std::array<Slice, inner_dim> inner_dims;
            std::array<Slice, outer_dim> outer_dims;
            return splitter<0, 0, inner_dim, 0, outer_dim, Args...>::split(std::move(it), std::move(inner_dims), std::move(outer_dims), std::move(dims));
        }
    };

  public:
    static constexpr std::size_t dimensions = dim;
    using type = T;
    using reference_type = Tref;
    using iterator_type = Iterator;
    template<std::size_t inner_dim>
    using split_type = View<T, inner_dim, Iterator, Tref>;

    constexpr View() = default;
    View(const View&) = delete;
    View(View&&) noexcept = default;
    View& operator=(View&&) noexcept = default;
    constexpr View(Iterator it_p, std::array<Slice, dim> dims_p) : it(std::move(it_p)), dims(std::move(dims_p)) {}

    template<typename... Args, typename detail::only_for_sizes<Args...>* = nullptr>
    explicit constexpr View(Iterator it_p, Args&&... args) : it(std::move(it_p)) {
        initialize_sizes<0>(std::forward<Args>(args)...);
    }

    template<typename... Args, typename detail::only_for_slices<Args...>* = nullptr>
    explicit constexpr View(Iterator it_p, Args&&... args) : it(std::move(it_p)) {
        initialize_slices<0>(std::forward<Args>(args)...);
    }

    template<bool... Args>
    SplitView<Split<Args...>::inner_dim, Split<Args...>::outer_dim> split() {
        return splitter_helper<Split<Args...>>::split(it, dims);
    }

    template<typename Splittype>
    SplitView<Splittype::inner_dim, Splittype::outer_dim> split() {
        return splitter_helper<Splittype>::split(it, dims);
    }

    template<bool... Args>
    constexpr SplitView<Split<Args...>::inner_dim, Split<Args...>::outer_dim> split() const {
        return splitter_helper<Split<Args...>>::split(it, dims);
    }

    template<typename Splittype>
    constexpr SplitView<Splittype::inner_dim, Splittype::outer_dim> split() const {
        return splitter_helper<Splittype>::split(it, dims);
    }

    template<typename Function>
    bool foreach_element(Function&& func) {
        iterator it_l = begin();
        for (; !it_l.ended(); ++it_l) {
            if (!detail::foreach_dim<0, dim, Tref>::template pass_parameters(it_l.pos(), std::forward<Function>(func), *it_l)) {
                return false;
            }
        }
        return true;
    }

    Iterator& data() { return it; }
    constexpr const Iterator& data() const { return it; }

    template<typename Function>
    void foreach_parallel(Function&& func) {
        const iterator bg = begin();
#pragma omp parallel for default(shared)
        for (std::size_t i = 0; i < bg.get_end_index(); ++i) {
            iterator it_l = bg + i;
            detail::foreach_dim<0, dim, Tref>::template pass_parameters_void(it_l.pos(), std::forward<Function>(func), *it_l);
        }
    }

    void swap_dims(std::size_t i, std::size_t j) {
        if (i >= dims.size() || j >= dims.size()) {
            throw std::out_of_range("index out of bounds");
        }
        std::swap(dims[i], dims[j]);
    }

    template<typename... Args>
    Tref operator()(Args&&... args) noexcept {
        return i_<0>(0, std::forward<Args>(args)...);
    }

    template<typename... Args>
    constexpr const Tref operator()(Args&&... args) const noexcept {
        return i_<0>(0, std::forward<Args>(args)...);
    }

    template<typename... Args>
    Tref at(Args&&... args) {
        return at_<0>(0, std::forward<Args>(args)...);
    }

    template<typename... Args>
    constexpr const Tref at(Args&&... args) const {
        return at_<0>(0, std::forward<Args>(args)...);
    }

    template<std::size_t c>
    constexpr const Slice& slice() const {
        static_assert(c < dim, "dimension index out of bounds");
        return std::get<c>(dims);
    }

    constexpr const Slice& slice(std::size_t i) const { return dims.at(i); }
    constexpr const std::array<Slice, dim>& slices() const { return dims; }

    template<std::size_t c>
    constexpr std::size_t size() const {
        static_assert(c < dim, "dimension index out of bounds");
        return std::get<c>(dims).size;
    }
    constexpr std::size_t size(std::size_t i) const { return dims.at(i).size; }

    constexpr std::size_t total_size() const { return detail::foreach_dim<0, dim>::total_size(dims); }
    Tref operator[](std::size_t i) { return it[i]; }
    constexpr const Tref operator[](std::size_t i) const { return it[i]; }

    void reset(const T& initial_value) { std::fill(iterator::begin(this), iterator::end(this), initial_value); }

    iterator begin() { return iterator::begin(this); }
    iterator end() { return iterator::end(this); }
    constexpr const_iterator begin() const { return const_iterator::begin(this); }
    constexpr const_iterator end() const { return const_iterator::end(this); }
};

}  // namespace nvector

#endif
