from program import Program
import sys
import argparse
from utils.unionfind import unionfind
from .wrapper import Wrapper
from commons.definitions import HexColor
from utils.hexutils import state_to_str,MoveConvert

EXE_NN_AGENT_NAME="/home/cgao3/PycharmProjects/nnhex/play/exec_nn_agent.py "
EXE_HEX_PATH="/home/cgao/benzene-vanilla/src/mohex/mohex "

def run_single_match(black_agent, white_agent, verbose=False):
    game=[]
    black_agent.sendCommand("clear_board")
    white_agent.sendCommand("clear_board")
    black_groups=unionfind()
    white_groups=unionfind()
    turn=HexColor.BLACK
    game_status=HexColor.EMPTY
    while game_status==HexColor.EMPTY:
        if turn==HexColor.BLACK:
            move = black_agent.genmove_black()
            if move == "resign":
                print("black resign")
                print(state_to_str(game))
                return 1
            white_agent.play_black(move)
        else:
            move=white_agent.genmove_white()
            if move=="resign":
                print("white resign")
                print(state_to_str(game))
                return 0
            black_agent.play_white(move)
        #imove=raw_move_to_int(move)
        imove=MoveConvert.rawMoveToIntMove(move)
        black_groups, white_groups = GameCheck.updateUF(game, black_groups, white_groups, imove, turn)
        #black_groups, white_groups=update_unionfind(imove, turn, game, black_groups, white_groups)
        #gamestatus=winner(black_groups,white_groups)
        gamestatus=GameCheck.winner(black_groups,white_groups)
        game.append(imove)
        if verbose:
            print(state_to_str(game))
        turn=HexColor.EMPTY-turn
        sys.stdout.flush()
    print("gamestatus", gamestatus)
    print(state_to_str(game))
    return gamestatus

if __name__ == "__main__":
    parser=argparse.ArgumentParser(description="tournament between agents")
    parser.add_argument("num_games", type=int, help="num of paired games playing")
    parser.add_argument("model_path", help="where the model locates", type=str)
    parser.add_argument("--wolve_path",default="/home/cgao3/benzene-vanilla/src/wolve/wolve", help="where is the wolve", type=str)
    parser.add_argument("--value_net", help="whether it is valuenet model", action="store_true", default=False)
    parser.add_argument("--verbose", help="verbose?", action="store_true", default=False)

    args=parser.parse_args()
    num_games=5000
    think_time=100
    net_exe=EXE_NN_AGENT_NAME + args.model_path +" 2>/dev/null"
    net_exe="/home/cgao/CLionProjects/cnn-benzene-play/cmake-build/src/mohex/mohex 2>/dev/null"
    #EXE_HEX_PATH=args.wolve_path
    bot_exe=EXE_HEX_PATH+" 2>/dev/null"
    bot=WrapperAgent(bot_exe, True)
    net=WrapperAgent(net_exe, True)
    white_win_count=0
    black_win_count=0
    for i in range(num_games):
        bot.reconnect()
        if EXE_HEX_PATH == args.wolve_path:
            bot.sendCommand("param_wolve max_time " + repr(think_time))
        else:
            bot.sendCommand("param_mohex max_time " + repr(think_time))
            bot.sendCommand("param_mohex num_threads 1")
            bot.sendCommand("param_mohex max_games 1000")

            net.sendCommand("param_mohex max_time "+ repr(think_time))
            net.sendCommand("param_mohex num_threads 1")
            net.sendCommand("param_mohex max_games 1000")
            net.sendCommand("boardsize " + repr(BOARD_SIZE))
        bot.sendCommand("boardsize "+ repr(BOARD_SIZE))
        win=run_single_match(bot, net, False)
        if win==HexColor.BLACK: black_win_count += 1
        if win==HexColor.WHITE: white_win_count +=1
        print(i+1, "black: ", black_win_count, "white: ", white_win_count)
    net.sendCommand("close")
    print("black win ", black_win_count, "white win count ", white_win_count)