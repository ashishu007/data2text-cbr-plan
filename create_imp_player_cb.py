import json, numpy as np
from datasets import load_dataset
from tqdm import tqdm
from utils import ExtractConceptOrder, ExtractEntities, GetEntityRepresentation
from sel_imp_players import get_imp_players_train, get_game_repr_imp_players

"""
Not right
Representation of game data.
Let's say we have a game with two teams (T1 & T2), and each team has 3 players ((P1, P2, P3) from T1 and (P4, P5, P6) from T2). Teams and players have their own attributes/features (e.g. PTS, REB, AST, STL, BLK).
Assuming that T2 won; and P1's PTS is higher than P2's PTS such that P1 > P2 > P3 in T1 and similarly P4 > P5 > P6 in T2, we can represent the game as:
[T2, T1, P4, P5, P6, P1, P2, P3, T2P4, T2P5, T2P6, T1P1, T1P2, T1P3, P4P5, P4P6, P5P6, P1P2, P1P3, P2P3]
[teams, players, teams|players, players|players]
[win_team, lose_team, win_players, lose_players, win_team|win_players, lose_team|lose_players, win_players*win_players, lose_players*lose_players]
"""

part = 'train'
dataset = load_dataset('GEM/sportsett_basketball')
sents_data = json.load(open(f'data/{part}_data_ct.json'))
eco_obj = ExtractConceptOrder()
ee_obj = ExtractEntities()
ger_obj = GetEntityRepresentation()

prob, sol = [], []
for idx, entry in tqdm(enumerate(dataset[f'{part}'])):
    entry_sents = list(filter(lambda x: x['gem_id'] == entry['gem_id'], sents_data))
    unique_summary_idx = set([x['summary_idx'] for x in entry_sents])
    for us_id in unique_summary_idx:
        summary_sents = list(filter(lambda x: x['summary_idx'] == us_id, entry_sents))
        sentences = [x['coref_sent'] for x in summary_sents]
        concept_order_w_ents = eco_obj.extract_concept_order(entry, summary_sents)
        concept_order = [x.split('|')[1] for x in concept_order_w_ents]
        imp_players = get_imp_players_train(entry, sentences, ee_obj)
        game_repr = get_game_repr_imp_players(entry, ger_obj, imp_players)
        prob.append(game_repr)
        sol.append(concept_order)
prob1 = np.array(prob)
print(f'{len(prob)} {len(sol)} {prob1.shape} entries in cb')
np.savez(f'cbs/imp_player_cb_prob.npz', prob1)
json.dump(sol, open('cbs/imp_player_cb_sol.json', 'w'), indent='\t')
