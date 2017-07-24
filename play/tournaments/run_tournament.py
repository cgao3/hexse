from program import Program
import sys
import argparse
from utils.unionfind import unionfind
from play.tournaments.wrapper import Wrapper
from commons.definitions import HexColor
from utils.hexutils import state_to_str,MoveConvert, GameCheck


def run_single_match(black_agent, white_agent, boardsize, opening='', verbose=False):
    game=[]
    black_agent.sendCommand("clear_board")
    white_agent.sendCommand("clear_board")
    black_groups=unionfind()
    white_groups=unionfind()
    turn=HexColor.BLACK
    if opening:
        int_move=MoveConvert.raw_move_to_int_move(opening, boardsize)
        game.append(int_move)
        turn=HexColor.WHITE
        white_agent.play_black(opening)
        black_agent.play_black(opening)
        black_groups, white_groups = GameCheck.updateUF(game, black_groups, white_groups, int_move, turn, boardsize)

    game_status=HexColor.EMPTY
    while game_status==HexColor.EMPTY:
        if turn==HexColor.BLACK:
            move = black_agent.genmove_black()
            if move == "resign":
                print("black resign")
                print(state_to_str(game, boardsize))
                return HexColor.WHITE
            white_agent.play_black(move)
        else:
            move=white_agent.genmove_white()
            if move=="resign":
                print("white resign")
                print(state_to_str(game, boardsize))
                return HexColor.BLACK
            black_agent.play_white(move)
        int_move=MoveConvert.raw_move_to_int_move(move, boardsize)
        black_groups, white_groups = GameCheck.updateUF(game, black_groups, white_groups, int_move, turn, boardsize)
        game_status=GameCheck.winner(black_groups,white_groups)
        game.append(int_move)
        if verbose:
            print(state_to_str(game, boardsize))
        turn=HexColor.EMPTY-turn
        sys.stdout.flush()
    print("gamestatus", game_status)
    print(state_to_str(game,boardsize))
    return game_status

def mohex_settings(agent):
    agent.sendCommand('param_mohex max_time 1')
    pass

def wolve_settings(agent):
    agent.sendCommand('param_wolve max_depth 1')
    #agent.sendCommand('param_wolve max_time 1')
    pass

if __name__ == "__main__":
    parser=argparse.ArgumentParser(description="tournament between agents")
    parser.add_argument("--num_games", type=int, default=10, help="num of paired games playing")
    parser.add_argument('--exe_black', type=str, default='', help='path to executable 1')
    parser.add_argument('--exe_white', type=str, default='', help='path to executable 2')
    parser.add_argument('--boardsize', type=int, default=9, help='board size to play on')
    parser.add_argument('--openings', type=str, default='', help='path to a file contains opening moves')
    parser.add_argument("--verbose", help="verbose?", action="store_true", default=False)
    args=parser.parse_args()

    num_games=args.num_games
    opening_moves=[]
    import os
    if os.path.isfile(args.openings):
        with open(args.openings, 'r') as f:
            for line in f:
                opening_moves.append(line.strip())

    black_player=Wrapper(args.exe_black, True)
    white_player=Wrapper(args.exe_white, True)
    black_win_count=0
    white_win_count=0

    for i in range(num_games):
        black_player.sendCommand('boardsize '+repr(args.boardsize))
        white_player.sendCommand("boardsize " + repr(args.boardsize))
        if 'mohex' in args.exe_black:
            mohex_settings(black_player)
        if 'mohex' in args.exe_white:
            mohex_settings(white_player)
        if 'wolve' in args.exe_black:
            wolve_settings(black_player)
        if 'wolve' in args.exe_white:
            wolve_settings(white_player)
        if not opening_moves:
            print('empty board opening')
            winner =run_single_match(black_player, white_player, boardsize=args.boardsize, opening='', verbose=False)
        else:
            k=i%len(opening_moves)
            print('opening move '+opening_moves[k])
            winner = run_single_match(black_player, white_player, boardsize=args.boardsize, opening=opening_moves[k], verbose=False)
        if winner == HexColor.BLACK: black_win_count += 1
        if winner == HexColor.WHITE: white_win_count +=1
        print(i+1, "black: ", black_win_count, "white: ", white_win_count)

    black_player.sendCommand('quit')
    white_player.sendCommand('quit')
    print("black win ", black_win_count, "white win count ", white_win_count)
    print('black is '+args.exe_black+' , white is '+args.exe_white)
