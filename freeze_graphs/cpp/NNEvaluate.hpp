#ifndef _NEURAL_H_
#define _NEURAL_H_
#include "tensorflow/core/public/session.h"
#include "Hex.hpp"
#include "Bitset.hpp"
#include "BitsetIterator.hpp"
#include <vector>

using namespace tensorflow;
class NNEvaluator {

public:
    Session *m_sess;
    size_t m_boardsize;
    size_t m_input_width;
    size_t m_input_padding;
    size_t m_input_depth;
    //int m_board[11][11];
    //std::vector<std::vector<int> > m_board;

public:
    NNEvaluator(std::string slmodel_path, int boardsize);
    ~NNEvaluator();
    void SetBoardSize(size_t size){
        m_boardsize = size;
    }
    void SetInputWidth(size_t width){
        m_input_width=width;
    }
    void SetInputPadding(size_t padding){
        m_input_padding=padding;
    }
    float evaluate(const benzene::bitset_t &black,
                   const benzene::bitset_t &white, int toplay, std::vector<float> &score) const;
    void make_input_tensor(const benzene::bitset_t &black_stones,
                           const benzene::bitset_t &white_stones, int toplay, Tensor& tensor) const;
};

#endif
