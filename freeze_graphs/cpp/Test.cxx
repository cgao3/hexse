#include "NNEvaluate.hpp"

std::string toString(size_t p, int boardsize){
    size_t x,y;
    x=p/boardsize;
    y=p%boardsize;
    char a=(char)('a'+x);
    std::string res;
    res.push_back(a);
    res.append(std::to_string(y));
    return res;
}

int main(){
    std::string slpath=std::string(ABS_TOP_SRCDIR)+"/share/models/const_graph.pb";

    NNEvaluator nn(slpath, 13);
    benzene::bitset_t b(std::string("0"));
    benzene::bitset_t w(std::string("0"));
    std::cout<<"ABS_TOP_SRC_DIR: "<<ABS_TOP_SRCDIR <<"\n";
    std::cout<<"slmodelpath: "<<slpath<<"\n\n";
    std::cout<<b<<"\n";
    std::vector<float> score;
    score.resize(13*13);
    b[7]=1;
    nn.evaluate(b,w,benzene::WHITE,score);
    std::cout<<"neural net output:";
    size_t best;
    float best_value=-0.1f;
    for(size_t i=0;i<score.size();i++){
        std::cout<<toString(i,13)<<":"<<score[i]<<"\n";
        if(best_value<score[i]){
            best=i;
            best_value=score[i];
        }
    }
    std::cout<<"\n";
    std::cout<<"best action is: "<<toString(best,13)<<"\n";
    return 0;
}