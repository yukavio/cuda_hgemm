// Copyright 2023. All Rights Reserved.
// Author: Bruce-Lee-LY
// Date: 20:42:28 on Sun, Feb 12, 2023
//
// Description: matrix

#pragma once

#include <random>

#include "common.h"

class Matrix {
public:
    Matrix(size_t row, size_t col, const std::string &name = "Matrix", float min = -1.0, float max = 1.0, int debug_flag=0)
        : m_row(row), m_col(col), m_name(name), m_min(min), m_max(max) {
        HGEMM_CHECK_GT(m_row, 0);
        HGEMM_CHECK_GT(m_col, 0);

        m_elem_num = m_row * m_col;
        HGEMM_CHECK_GT(m_elem_num, 0);

        m_host_ptr = new half[m_elem_num];
        HGEMM_CHECK(m_host_ptr);
        HGEMM_CHECK_CUDART_ERROR(cudaMalloc((void **)&m_dev_ptr, m_elem_num * sizeof(half)));
        HGEMM_CHECK(m_dev_ptr);

        std::random_device rd;
        std::default_random_engine engine{rd()};
        std::uniform_real_distribution<float> uniform(m_min, m_max);
        if (debug_flag == 0){
            for (size_t i = 0; i < m_elem_num; ++i) {
                m_host_ptr[i] = __float2half(uniform(engine));
            }
        } else if(debug_flag == 1){
            for(size_t i=0; i < m_row; i++){
                for(size_t j=0; j<m_col; j++){
                    m_host_ptr[i*m_row+j] = i==j ? __float2half(1.0) : __float2half(0.0);
                }
            }
        }
        else if(debug_flag == 2){
            for (size_t i = 0; i < m_elem_num; ++i) {
                m_host_ptr[i] = __float2half(i*1.0);
            }
        }

        HGEMM_CHECK_CUDART_ERROR(cudaMemcpy(m_dev_ptr, m_host_ptr, m_elem_num * sizeof(half), cudaMemcpyHostToDevice));

        HLOG("%s: %zu * %zu, cpu: %p, gpu: %p", m_name.c_str(), m_row, m_col, m_host_ptr, m_dev_ptr);
    }

    ~Matrix() {
        if (m_host_ptr) {
            delete[] m_host_ptr;
            m_host_ptr = nullptr;
        }

        if (m_dev_ptr) {
            HGEMM_CHECK_CUDART_ERROR(cudaFree((void *)m_dev_ptr));
            m_dev_ptr = nullptr;
        }
    }

    size_t getRow() const {
        return m_row;
    }

    size_t getCol() const {
        return m_col;
    }

    size_t getElemNum() const {
        return m_elem_num;
    }

    half *getHostPtr() const {
        return m_host_ptr;
    }

    half *getDevPtr() const {
        return m_dev_ptr;
    }

    void tearUp(Matrix *base) {
        HGEMM_CHECK(base);
        HGEMM_CHECK_EQ(m_row, base->getRow());
        HGEMM_CHECK_EQ(m_col, base->getCol());

        HGEMM_CHECK_CUDART_ERROR(
            cudaMemcpy(m_dev_ptr, base->getDevPtr(), m_elem_num * sizeof(half), cudaMemcpyDeviceToDevice));
    }

    void moveToHost() {
        HGEMM_CHECK_CUDART_ERROR(cudaMemcpy(m_host_ptr, m_dev_ptr, m_elem_num * sizeof(half), cudaMemcpyDeviceToHost));
    }

    void moveToDevice() {
        HGEMM_CHECK_CUDART_ERROR(cudaMemcpy(m_dev_ptr, m_host_ptr, m_elem_num * sizeof(half), cudaMemcpyHostToDevice));
    }

    void memSetHost() {
        memset(m_host_ptr, 0, m_elem_num * sizeof(half));
    }

    void memSetDevice() {
        HGEMM_CHECK_CUDART_ERROR(cudaMemset(m_dev_ptr, 0, m_elem_num * sizeof(half)));
    }


    void checkValue_branch(Matrix *base, int branch) {
        HGEMM_CHECK(base);
        HGEMM_CHECK_EQ(m_row, base->getRow());
        HGEMM_CHECK_EQ(m_col, base->getCol());

        m_max_diff = 0.0;
        m_avg_diff = 0.0;
        double diff = 0.0;
        for (size_t i = 0; i < m_row; ++i) {
            for(size_t j=0; j< m_col/branch; j++){
                float ref_value = 0.0;
                for(int t=0; t<branch; t++){
                    ref_value += __half2float(base->getHostPtr()[(i*m_col/branch+j)*branch+t]);
                }
                //printf("%f|%f ", __half2float(m_host_ptr[i*m_col/branch+j]), ref_value);
                diff = static_cast<double>(std::abs(__half2float(m_host_ptr[i*m_col/branch+j]) - ref_value));
                m_max_diff = std::max(m_max_diff, diff);
                m_avg_diff += diff;
            }
            //printf("\n");
        }

        m_avg_diff /= static_cast<double>(m_elem_num);

        HLOG("Max diff: %f, avg diff: %f", m_max_diff, m_avg_diff);
    }
        
    void show(int branch){
        for(int i=0; i < m_row; i++){
            for(int j=0; j<m_col/branch; j++){
                float ref_value = 0.0;
                for(int t=0; t<branch; t++){
                    ref_value += __half2float(getHostPtr()[(i*m_col/branch+j)*branch+t]);
                }
                printf("%f ", ref_value);
            }
            printf("\n");
        }
    }

    void show_res(int branch){
        for(int i=0; i < m_row; i++){
            for(int j=0; j<m_col/branch; j++){
                printf("%f ", __half2float(getHostPtr()[i*m_col/branch+j]));
            }
            printf("\n");
        }
    }

    void checkValue(Matrix *base) {
        HGEMM_CHECK(base);
        HGEMM_CHECK_EQ(m_row, base->getRow());
        HGEMM_CHECK_EQ(m_col, base->getCol());

        m_max_diff = 0.0;
        m_avg_diff = 0.0;
        double diff = 0.0;
        for (size_t i = 0; i < m_elem_num; ++i) {
            diff = static_cast<double>(std::abs(__half2float(m_host_ptr[i]) - __half2float(base->getHostPtr()[i])));
            m_max_diff = std::max(m_max_diff, diff);
            m_avg_diff += diff;
        }

        m_avg_diff /= static_cast<double>(m_elem_num);

        HLOG("Max diff: %f, avg diff: %f", m_max_diff, m_avg_diff);
    }

private:
    const size_t m_row = 0;
    const size_t m_col = 0;
    const std::string m_name = "Matrix";
    // the threshold of the random matrix will affect the difference of the hgemm results
    const float m_min = -1.0;
    const float m_max = 1.0;

    size_t m_elem_num = 0;
    half *m_host_ptr = nullptr;
    half *m_dev_ptr = nullptr;

    double m_max_diff = 0.0;
    double m_avg_diff = 0.0;

    HGEMM_DISALLOW_COPY_AND_ASSIGN(Matrix);
};

