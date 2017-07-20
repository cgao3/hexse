#include "tensorflow/core/public/session.h"
#include "tensorflow/core/platform/env.h"
#include<iostream>
#include<vector>
#include<utility>

#include "NNEvaluate.hpp"

NNEvaluator::~NNEvaluator(){
    m_sess->Close();
}

NNEvaluator::NNEvaluator(std::string slmodel_path, int b){
    m_boardsize= static_cast<size_t>(b);
    m_input_padding=2;
    m_input_width=m_boardsize+2*m_input_padding;
    m_input_depth=9;



    Status status = NewSession(SessionOptions(), &this->m_sess);
    if (!status.ok()) {
        std::cout << "something wrong while creating slmodel session, " <<status.ToString() << "\n";
    }

    GraphDef graph_def;
    status = ReadBinaryProto(Env::Default(), slmodel_path, &graph_def);
    if (!status.ok()) {
        std::cout <<"something wrong reading sl graph"<< status.ToString() << "\n";
    }

    // Add the graph to the session
    status = this->m_sess->Create(graph_def);
    if (!status.ok()) {
        std::cout << status.ToString() << "\n";
    }

}

//toplay actually useless
void NNEvaluator::make_input_tensor(const benzene::bitset_t &black_stones,
                                    const benzene::bitset_t &white_stones, int toplay, Tensor &input) const {

    //input depth=9
    int BlackStones=0;
    int WhiteStones=1;
    int BlackBridgeEndPoints=2;
    int WhiteBridgeEndPoints=3;
    int BlackToPlay=4;
    int WhiteToPlay=5;
    int ToPlaySaveBridge=6;
    int ToPlayFormBridge=7;
    int ToPlayEmptyPoints=8;

    size_t i,j, k;
    int x,y;

    std::vector< std::vector<int> > m_board;

    m_board.resize(m_input_width);
    for(size_t i=0; i<m_board.size(); i+=1){
        m_board[i].resize(m_input_width);
        std::fill(m_board[i].begin(), m_board[i].end(), benzene::EMPTY);
    }
    auto ten=input.tensor<float,4>();

    //set the empty points
    for (i=0;i<m_input_width;i++){
        for(j=0;j<m_input_width;j++){
            for(k=0;k<m_input_depth; k++){
                ten(0,i,j,k)=0;
                if(k==ToPlayEmptyPoints && i>=m_input_padding && j>=m_input_padding
                        && i<m_input_width-m_input_padding && j<m_input_width-m_input_padding){
                    ten(0,i,j,k)=1;
                }
            }
        }
    }

    //black stones
    for(i=0;i<m_input_padding;i++){
        for(j=0;j<m_input_width;j++){
            ten(0,j,i,BlackStones)=1;
            ten(0,j,m_input_width-1-i,BlackStones)=1;
        }
    }
    //white stones
    for(i=0;i<m_input_padding;i++){
        for(j=m_input_padding;j<m_input_width-m_input_padding;j++){
            ten(0,i,j,WhiteStones)=1;
            ten(0,m_input_width-1-i,j,WhiteStones)=1;
        }
    }

    //set m_board, and black/white played stones, modify empty points
    for (benzene::BitsetIterator it(black_stones); it; ++it){
        int p=*it-7;
        if (p<0) continue;
        benzene::HexPointUtil::pointToCoords(*it, x,y);
        x += m_input_padding;
        y += m_input_padding;
        m_board[x][y]=benzene::BLACK;
        ten(0,x,y,BlackStones)=1.0;
        ten(0,x,y,ToPlayEmptyPoints)=0.0;
    }
    for (benzene::BitsetIterator it(white_stones); it; ++it){
        int p=*it-7;
        if(p<0) continue;
        benzene::HexPointUtil::pointToCoords(*it,x,y);
        x += m_input_padding;
        y += m_input_padding;
        m_board[x][y]=benzene::WHITE;
        ten(0,x,y,WhiteStones)=1.0;
        ten(0,x,y,ToPlayEmptyPoints)=0.0;
    }

    //set up toplay plane
    if(toplay==benzene::BLACK){
        for(i=0;i<m_input_width;i++){
            for(j=0;j<m_input_width;j++){
                ten(0,i,j,BlackToPlay)=1.0;
                ten(0,i,j,WhiteToPlay)=0.0;
            }
        }
    } else if(toplay==benzene::WHITE){
        for(i=0;i<m_input_width;i++){
            for(j=0;j<m_input_width;j++){
                ten(0,i,j,WhiteToPlay)=1.0;
                ten(0,i,j,BlackToPlay)=0.0;
            }
        }
    } else {
        std::cout<<"ERROR TOPLAY "<<toplay<<"\n";
    }
    int B=benzene::BLACK, W=benzene::WHITE, E=benzene::EMPTY;
    for(i=m_input_padding;i<m_input_width-m_input_padding;i++){
        for(j=0;j<m_input_width-m_input_padding;j++){
            int point0, point1, point2, point3;
            point0=m_board[i][j];
            point1=m_board[i+1][j];
            point2=m_board[i][j+1];
            point3=m_board[i+1][j+1];
            //bridge endpoints
            if(point0==B && point3==B && point1!=W && point2!=W){
                ten(0,i,j,BlackBridgeEndPoints)=1.0;
                ten(0,i+1,j+1,BlackBridgeEndPoints)=1.0;
            }
            else if(point0==W && point3==W && point1!=B && point2!=B){
                ten(0,i,j,WhiteBridgeEndPoints)=1.0;
                ten(0,i+1,j+1,WhiteBridgeEndPoints)=1.0;
            }
            if(j>=1){
                int q0,q1,q2,q3;
                q0=m_board[i][j];
                q1=m_board[i+1][j-1];
                q2=m_board[i+1][j];
                q3=m_board[i][j+1];
                if(q0==B && q3==B && q1!=W && q2!=W){
                    ten(0,i+1,j-1,BlackBridgeEndPoints)=1.0;
                    ten(0,i,j+1,BlackBridgeEndPoints)=1.0;
                } else if(q0==W && q3==W && q1!=B && q2!=B){
                    ten(0,i+1,j-1,WhiteBridgeEndPoints)=1.0;
                    ten(0,i,j+1,WhiteBridgeEndPoints)=1.0;
                }
            }

            //toplay save bridge
            if(point0==point3 && point0==toplay &&
                    point1!=toplay && point2!=toplay){
                if(point1==E && point2!=E){
                    ten(0,i+1,j,ToPlaySaveBridge)=1.0;
                } else if(point1!=E && point2==E){
                    ten(0,i,j+1,ToPlaySaveBridge)=1.0;
                }
            }
            if(j>=1){
                int q0,q1,q2,q3;
                q0=m_board[i][j];
                q1=m_board[i+1][j-1];
                q2=m_board[i+1][j];
                q3=m_board[i][j+1];
                if(q1==q3 && q1==toplay && q0!=toplay && q2!=toplay){
                    if(q0==E && q2!=E){
                        ten(0,i,j,ToPlaySaveBridge)=1.0;
                    } else if(q2!=E && q1==E){
                        ten(0,i+1,j,ToPlaySaveBridge)=1.0;
                    }
                }
            }

            //toplay form bridge
            if(point1==E && point2==E){
                if(point0==E && point3==toplay){
                    ten(0,i,j,ToPlayFormBridge)=1.0;
                } else if(point0==toplay && point3==E){
                    ten(0,i+1,j+1,ToPlayFormBridge)=1.0;
                }
            }
            if(j>=1){
                int q0,q1,q2,q3;
                q0=m_board[i][j];
                q1=m_board[i+1][j-1];
                q2=m_board[i+1][j];
                q3=m_board[i][j+1];
                if(q0==E && q2==E){
                    if(q1==toplay && q3==E){
                        ten(0,i,j+1,ToPlayFormBridge)=1.0;
                    } else if(q1==E && q3==toplay){
                        ten(0,i+1,j-1,ToPlayFormBridge)=1.0;
                    }
                }
            }
        }
    }
}


float NNEvaluator::evaluate(const benzene::bitset_t &black, const benzene::bitset_t &white,
                            int toplay, std::vector<float> &score) const {

    auto t1=std::chrono::system_clock::now();
    Tensor input(DT_FLOAT, TensorShape({1, static_cast<int64 >(m_input_width),
                                        static_cast<int64 >(m_input_width), 9}));
    //std::cout<<"here is normal\n";
    make_input_tensor(black, white, toplay, input);
    //std::cout<<"normal 2\n";
    std::vector<std::pair<string, tensorflow::Tensor>> inputs={
            {"x_input_node", input},
    };

    std::vector<tensorflow::Tensor> output;
    Status status=this->m_sess->Run(inputs, {"ConvLayer7_3/output_node"},{}, &output);
    if(!status.ok()){
        std::cout<<"error:"<<status.ToString()<<"\n";
    }
    auto ret=output[0].flat<float>();
    float sum_value=0.0;

    /*
     * the score vector contains all logits for each move, whether valid or invalid
     * Note that indices of score should be converted into normal move
     * by x=i//boardsize, y=i%boardsize
     * where x is then converted into x+'a'
     * y -> y+1
     *
     * This conversion is different by what has been adopted by Benzene!
     */

    for(int i=0;i<m_boardsize*m_boardsize;i++){
        score[i]=ret(i);
        score[i]=(float)exp(score[i]);
        sum_value +=score[i];
    }

    for(int i=0;i<m_boardsize*m_boardsize;i++){
        score[i]=score[i]/sum_value;
    }

    auto t2=std::chrono::system_clock::now();
 //   std::cout<<"time cost per eva:"<<std::chrono::duration_cast<std::chrono::microseconds>(t2-t1).count()/1e06<<" seconds\n";

    return 0.2;
}
