/**
 * @file matrix.hpp
 * @author 手写AI (zifuture.com:8090)
 * @brief 矩阵的简单实现类。现写的，调用OpenBLAS实现高性能的矩阵乘法
 * @date 2021-08-03
 * 
 * @copyright Copyright (c) 2021
 */

#ifndef MATRIX_HPP
#define MATRIX_HPP

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <vector>
#include <random>
#include <memory>
#include <openblas/cblas.h>

// 打印信息时带上文件和行号
#define INFO(...)                                \
    do{                                          \
        printf("[%s:%d]: ", __FILE__, __LINE__); \
        printf(__VA_ARGS__);                     \
        printf("\n");                            \
    }while(0)

// 断言的定义
#define Assert(cond)                                    \
    do{                                                 \
        bool op = !(!(cond));                           \
        if(!op){                                        \
            INFO("Assert failed, op = %s\n", #cond);    \
            abort();                                    \
        }                                               \
    }while(0)

// 把大端的整数变为小端的整数/把小端的整数变为大端的整数
#define InvertBit(number)           (((number & 0x000000FF) << 24)  |   \
                                     ((number & 0x0000FF00) << 8)   |   \
                                     ((number & 0x00FF0000) >> 8)   |   \
                                     ((number & 0xFF000000) >> 24))

// 定义矩阵操作的基本类，根据_DataType决定储存的数据类型
// Matrix的数据是引用关系的，即a = b，数据公用一份，原因是 shared_ptr
// 如果希望复制，请使用copy()函数
template<typename _DataType>
class Matrix{
public:
    Matrix() = default;
    Matrix(int rows, int cols, const _DataType* pdata = nullptr){
        resize(rows, cols);
        if(pdata)
            memcpy(data_->data(), pdata, rows * cols * sizeof(_DataType));
    }

    const _DataType* ptr(int row_index = 0) const {return data_->data() + row_index * cols_;}
          _DataType* ptr(int row_index = 0)       {return data_->data() + row_index * cols_;}
    int rows()   const{return rows_;}
    int cols()   const{return cols_;}
    int numel()  const{return numel_;}
    bool empty() const{return data_ == nullptr;}

    Matrix<_DataType> copy() const {
        Matrix<_DataType> out = *this;
        out.data_ = std::make_shared<std::vector<_DataType>>(*this->data_);
        return out;
    }

    void resize(int rows, int cols){
        rows_ = rows;
        cols_ = cols;
        numel_ = rows_ * cols_;

        if(data_ == nullptr || numel_ != data_->size()){
            data_ = std::make_shared<std::vector<_DataType>>();
            data_->resize(numel_);
        }
    }

    // 定义矩阵的转置，行列颠倒
    Matrix<_DataType> T(){

        Matrix<_DataType> out(cols_, rows_);
        for(int i = 0; i < rows_; ++i){
            auto row_ptr = ptr(i);
            auto out_ptr = out.ptr() + i;
            for(int j = 0; j < cols_; ++j, out_ptr += out.cols())
                *out_ptr = row_ptr[j];
        }
        return out;
    }

    Matrix<_DataType> sigmoid(){
        auto out = this->copy();
        return out.sigmoid_();
    }
    
    // inplace的sigmoid
    Matrix<_DataType>& sigmoid_(){
        auto ptr = data_->data();
        for(int i = 0; i < data_->size(); ++i, ++ptr){
            _DataType& x = *ptr;

            // 避免sigmoid中exp的上溢出
            if(x < 0){
                x = exp(x) / (1 + exp(x));
            }else{
                x = 1 / (1 + exp(-x));
            }
        }
        return *this;
    }

    Matrix<_DataType> relu(){
        auto out = this->copy();
        return out.relu_();
    }
    
    // inplace的sigmoid
    Matrix<_DataType>& relu_(){
        auto ptr = data_->data();
        for(int i = 0; i < data_->size(); ++i, ++ptr)
            *ptr = std::max<_DataType>(_DataType(0), *ptr);
        return *this;
    }

private:
    std::shared_ptr<std::vector<_DataType>> data_;
    int rows_ = 0, cols_ = 0, numel_ = 0;
};

typedef Matrix<float>         Matrixf;
typedef Matrix<unsigned char> Matrixu;

/**
 * @brief 定义矩阵的ab操作，即 c = a op b
 *        例如c[Matrix] = a[Matrix] + b[Matrix]，或者
 *            c[Matrix] = a[Number] + b[Matrix]
 * 
 * 这里有以下几类：
 * 1. 广播机制：c[Matrix, mxn] = a[Matrix, 1xn] + b[Matrix, mxn]
 * 2. 元素操作：c[Matrix, mxn] = a[Matrix, mxn] - b[Matrix, mxn]
 * 3. 左值操作：c[Matrix, mxn] = a[Number]      * b[Matrix, mxn]
 * 4. 右值操作：c[Matrix, mxn] = a[Matrix, mxn] / b[Number]
 */
#define define_matrix_ab_op(name, op)                                          \
    template<typename _TA, typename _TB>                                        \
    Matrix<_TA> operator op (const Matrix<_TA>& a, const Matrix<_TB>& b);       \
                                                                                \
    template<typename _TA, typename _TB>                                        \
    Matrix<_TA> broadcast_##name (const Matrix<_TA>& a, const Matrix<_TB>& b){  \
        if(a.rows() == 1 && a.cols() == b.cols()){                              \
            auto out = b.copy();                                                \
            auto aptr = a.ptr();                                                \
            auto optr = out.ptr();                                              \
            for(int i = 0; i < b.numel(); ++i, ++optr){                         \
                *optr = aptr[i % b.cols()] op *optr;                            \
            }                                                                   \
            return out;                                                         \
        }else if(a.cols() == 1 && a.rows() == b.rows()){                        \
            auto out = b.copy();                                                \
            auto aptr = a.ptr();                                                \
            auto optr = out.ptr();                                              \
            for(int i = 0; i < b.numel(); ++i, ++optr){                         \
                *optr = aptr[i / b.cols()] op *optr;                            \
            }                                                                   \
            return out;                                                         \
        }else if(b.rows() == 1 && b.cols() == a.cols() || b.cols() == 1 && b.rows() == a.rows()){   \
            return broadcast_##name(b, a);                                      \
        }else{                                                                  \
            Assert(a.rows() == b.rows() && a.cols() == b.cols());               \
            return a op b;                                                      \
        }                                                                       \
    }                                                                           \
                                                                                \
    template<typename _TA, typename _TB>                                        \
    Matrix<_TA> operator op (const Matrix<_TA>& a, const Matrix<_TB>& b){       \
        if(std::min(a.rows(), b.rows()) == 1 && a.rows() != b.rows() ||         \
           std::min(a.cols(), b.cols()) == 1 && a.cols() != b.cols()            \
        ){                                                                      \
            return broadcast_##name(a, b);                                      \
        }                                                                       \
                                                                                \
        Assert(a.rows() == b.rows() and a.cols() == b.cols());                  \
        auto c = a.copy();                                                      \
        auto ptra = a.ptr();                                                    \
        auto ptrb = b.ptr();                                                    \
        auto ptrc = c.ptr();                                                    \
        for(int i = 0; i < a.numel(); ++i, ++ptra, ++ptrb, ++ptrc){             \
            *ptrc = (*ptra) op (*ptrb);                                         \
        }                                                                       \
        return c;                                                               \
    }                                                                           \
                                                                                \
    template<typename _TA, typename _TB>                                        \
    Matrix<_TB> operator op (_TA a, const Matrix<_TB>& b){                      \
        auto out = b.copy();                                                    \
        auto ptr = out.ptr();                                                   \
        for(int i = 0; i < out.numel(); ++i, ++ptr)                             \
            *ptr = a op *ptr;                                                   \
        return out;                                                             \
    }                                                                           \
                                                                                \
    template<typename _TA, typename _TB>                                        \
    Matrix<_TA> operator op (const Matrix<_TA>& a, _TB b){                      \
        auto out = a.copy();                                                    \
        auto ptr = out.ptr();                                                   \
        for(int i = 0; i < out.numel(); ++i, ++ptr)                             \
            *ptr = *ptr op b;                                                   \
        return out;                                                             \
    }


// 通过宏定义，实现加减乘除的批量定义
define_matrix_ab_op(mul, *);
define_matrix_ab_op(add, +);
define_matrix_ab_op(sub, -);
define_matrix_ab_op(div, /);


/**
 * @brief 实现矩阵乘法
 * 这个版本效率没有优化，因此建议使用基于openblas的版本
 */
template<typename _TA, typename _TB>
Matrix<_TA> gemm_mul(const Matrix<_TA>& a, const Matrix<_TB>& b){

    // NMK,  a.shape = N x M, b.shape = M x K.  c.shape = N x K
    Assert(a.cols() == b.rows());

    Matrix<_TA> c(a.rows(), b.cols());

    #pragma omp parallel for
    for(int ir = 0; ir < c.rows(); ++ir){
        auto c_row_ptr = c.ptr(ir);
        auto a_row_ptr = a.ptr(ir);
        for(int ic = 0; ic < c.cols(); ++ic){
            _TA value = 0;
            for(int n = 0; n < a.cols(); ++n)
                value += a_row_ptr[n] * b.ptr(n)[ic];
            
            c_row_ptr[ic] = value;
        }
    }
    return c;
}

/**
 * @brief 实现矩阵乘法，调用OpenBLAS
 * 
 * @param a    矩阵A
 * @param ta   矩阵A是否转置
 * @param b    矩阵B
 * @param tb   矩阵B是否转置
 * @return Matrix<_TA>    返回 c = ta(a) @ tb(b)
 */
template<typename _TA, typename _TB>
Matrix<_TA> gemm_mul(const Matrix<_TA>& a, bool ta, const Matrix<_TB>& b, bool tb){

    // NMK,  a.shape = N x M, b.shape = M x K.  c.shape = N x K
    // 定义a和b，经过ta、tb后的shape
    int a_elastic_rows = ta ? a.cols() : a.rows();
    int a_elastic_cols = ta ? a.rows() : a.cols();
    int b_elastic_rows = tb ? b.cols() : b.rows();
    int b_elastic_cols = tb ? b.rows() : b.cols();
    Matrix<_TA> c(a_elastic_rows, b_elastic_cols);

    cblas_sgemm(
        CblasRowMajor,
        ta ? CblasTrans : CblasNoTrans, 
        tb ? CblasTrans : CblasNoTrans, 
        a_elastic_rows,
        b_elastic_cols,
        a_elastic_cols,
        1.0f,
        a.ptr(),
        a.cols(),
        b.ptr(),
        b.cols(),
        0.0f,
        c.ptr(),
        c.cols()
    );
    return c;
}

/**
 * @brief 打印矩阵
 * 
 * @param a       指定需要打印的矩阵
 * @param format  指定打印是的格式化方式。默认是float = %.3f，其他%3d
 */
template<typename _T>
void print_matrix(const Matrix<_T>& a, const char* format =
    std::__is_floating<_T>::__value ? "%.3f" : "%3d"
){
    INFO("Matrix[%p], %d x %d", &a, a.rows(), a.cols());

    char fmt[20];
    sprintf(fmt, "%s,", format);

    for(int i = 0; i < a.rows(); ++i){

        printf("row[%02d]: ", i);
        for(int j = 0; j < a.cols(); ++j){
            printf(fmt, a.ptr(i)[j]);
        }
        printf("\n");
    }
}


enum class TextColor : int{
    None   = 0,     // 无颜色配置
    Black  = 30,    // 黑色
    Red    = 31,    // 红色
    Green  = 32,    // 绿色
    Yellow = 33,    // 黄色
    Blue   = 34,    // 蓝色
    Rosein = 35,    // 品红
    Cyan   = 36,    // 青色
    White  = 37     // 白色
};

enum class TextBackgroundColor : int{
    None   = 0,     // 无颜色配置
    Black  = 40,    // 黑色
    Red    = 41,    // 红色
    Green  = 42,    // 绿色
    Yellow = 43,    // 黄色
    Blue   = 44,    // 蓝色
    Rosein = 45,    // 品红
    Cyan   = 46,    // 青色
    White  = 47     // 白色
};

static std::string color_text(
    const std::string& text, 
    TextColor color=TextColor::None, 
    TextBackgroundColor background=TextBackgroundColor::None
){
    if(color == TextColor::None && background == TextBackgroundColor::None)
        return text;

    #ifdef _WIN64
        // windows is ignore
        return text;
    #endif

    char line_text[1000] = "\033[";
    int size = sizeof(line_text) - 2;
    char* p  = line_text + 2;

    if(color != TextColor::None){
        const char* fmt = background != TextBackgroundColor::None ? "%d;" : "%d";
        int n = snprintf(p, size, fmt, color);
        p += n;  size -= n;
    }

    if(background != TextBackgroundColor::None){
        int n = snprintf(p, size, "%d", background);
        p += n;  size -= n;
    }
    snprintf(p, size, "m%s\033[0m", text.c_str());
    return line_text;
}

#endif // MATRIX_HPP