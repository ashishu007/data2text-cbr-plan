import json
import numpy as np
from datasets import load_dataset
from tqdm import tqdm
from utils import GetGameRepresentation

"""
Representation of game data.
Let's say we have a game with two teams (T1 & T2), and each team has 3 players ((P1, P2, P3) from T1 and (P4, P5, P6) from T2). Teams and players have their own attributes/features (e.g. PTS, REB, AST, STL, BLK).
Assuming that T2 won; and P1's PTS is higher than P2's PTS such that P1 > P2 > P3 in T1 and similarly P4 > P5 > P6 in T2, we can represent the game as:
[T2, T1, P4, P5, P6, P1, P2, P3, T2P4, T2P5, T2P6, T1P1, T1P2, T1P3, P4P5, P4P6, P5P6, P1P2, P1P3, P2P3]
[teams, players, teams|players, players|players]
[win_team, lose_team, win_players, lose_players, win_team|win_players, lose_team|lose_players, win_players*win_players, lose_players*lose_players]
"""

dataset = load_dataset('GEM/sportsett_basketball')
order_cb_only_sol = json.load(open('data/stage1_case_base.json'))
game_rep_obj = GetGameRepresentation()

ordering_cb_reps = []
# for idx, entry in tqdm(enumerate(dataset['train'])):
for idx, entry in tqdm(enumerate(dataset['validation'])):
    # if int(entry['gem_id'].split('-')[-1]) > 2460:
    cb_sols = list(filter(lambda x: x['problem_id'] == entry['gem_id'], order_cb_only_sol))
    for sol in cb_sols:
        ordering_cb_reps.append({"problem": game_rep_obj.get_full_game_repr(entry), "solution": sol['solution']})

print(f'{len(ordering_cb_reps)} entries in ordering_cb_reps')
json.dump(ordering_cb_reps, open('cbs/cb_stage1.json', 'w'), indent='\t')
