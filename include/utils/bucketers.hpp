#pragma once

#include "util.hpp"

namespace pthash {

struct skew_bucketer {
    skew_bucketer() {}

    void init(uint64_t num_buckets) {
        m_num_dense_buckets1 = constants::b1 * num_buckets;
        m_num_dense_buckets2 = constants::b2 * num_buckets;
        m_num_sparse_buckets = num_buckets - m_num_dense_buckets1 - m_num_dense_buckets2;
        m_M_num_dense_buckets1 = fastmod::computeM_u64(m_num_dense_buckets1);
        m_M_num_dense_buckets2 = fastmod::computeM_u64(m_num_dense_buckets2);
        m_M_num_sparse_buckets = fastmod::computeM_u64(m_num_sparse_buckets);
    }

    inline uint64_t bucket(uint64_t hash) const {
        // static const uint64_t T = constants::a * UINT64_MAX;
        // return (hash < T) ? fastmod::fastmod_u64(hash, m_M_num_dense_buckets,
        // m_num_dense_buckets)
        //                   : m_num_dense_buckets + fastmod::fastmod_u64(hash,
        //                   m_M_num_sparse_buckets,
        //                                                                m_num_sparse_buckets);
        static const uint64_t T1 = constants::a1 * UINT64_MAX;
        static const uint64_t T2 = (constants::a1 + constants::a2) * UINT64_MAX;
        if (hash < T1) {
            return fastmod::fastmod_u64(hash, m_M_num_dense_buckets1, m_num_dense_buckets1);
        }
        if (hash < T2) {
            return m_num_dense_buckets1 +
                   fastmod::fastmod_u64(hash, m_M_num_dense_buckets2, m_num_dense_buckets2);
        }
        return m_num_dense_buckets1 + m_num_dense_buckets2 +
               fastmod::fastmod_u64(hash, m_M_num_sparse_buckets, m_num_sparse_buckets);
    }

    uint64_t num_buckets() const {
        return m_num_dense_buckets1 + m_num_dense_buckets2 + m_num_sparse_buckets;
    }

    size_t num_bits() const {
        return 8 * (sizeof(m_num_dense_buckets1) + sizeof(m_num_dense_buckets2) +
                    sizeof(m_num_sparse_buckets) + sizeof(m_M_num_dense_buckets1) +
                    sizeof(m_M_num_dense_buckets2) + sizeof(m_M_num_sparse_buckets));
    }

    void swap(skew_bucketer& other) {
        std::swap(m_num_dense_buckets1, other.m_num_dense_buckets1);
        std::swap(m_num_dense_buckets2, other.m_num_dense_buckets2);
        std::swap(m_num_sparse_buckets, other.m_num_sparse_buckets);
        std::swap(m_M_num_dense_buckets1, other.m_M_num_dense_buckets1);
        std::swap(m_M_num_dense_buckets2, other.m_M_num_dense_buckets2);
        std::swap(m_M_num_sparse_buckets, other.m_M_num_sparse_buckets);
    }

    template <typename Visitor>
    void visit(Visitor& visitor) {
        visitor.visit(m_num_dense_buckets1);
        visitor.visit(m_num_dense_buckets2);
        visitor.visit(m_num_sparse_buckets);
        visitor.visit(m_M_num_dense_buckets1);
        visitor.visit(m_M_num_dense_buckets2);
        visitor.visit(m_M_num_sparse_buckets);
    }

private:
    uint64_t m_num_dense_buckets1, m_num_dense_buckets2, m_num_sparse_buckets;
    __uint128_t m_M_num_dense_buckets1, m_M_num_dense_buckets2, m_M_num_sparse_buckets;
};

struct uniform_bucketer {
    uniform_bucketer() {}

    void init(uint64_t num_buckets) {
        m_num_buckets = num_buckets;
        m_M_num_buckets = fastmod::computeM_u64(m_num_buckets);
    }

    inline uint64_t bucket(uint64_t hash) const {
        return fastmod::fastmod_u64(hash, m_M_num_buckets, m_num_buckets);
    }

    uint64_t num_buckets() const {
        return m_num_buckets;
    }

    size_t num_bits() const {
        return 8 * (sizeof(m_num_buckets) + sizeof(m_M_num_buckets));
    }

    template <typename Visitor>
    void visit(Visitor& visitor) {
        visitor.visit(m_num_buckets);
        visitor.visit(m_M_num_buckets);
    }

private:
    uint64_t m_num_buckets;
    __uint128_t m_M_num_buckets;
};

}  // namespace pthash