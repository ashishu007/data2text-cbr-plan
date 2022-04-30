import json
import numpy as np
from tqdm import tqdm
from sklearn.metrics.pairwise import euclidean_distances
from datasets import load_dataset
# from non_rg import NonRGMetrics
from entity_ranking import RankEntities
from utils import ExtractConceptOrder, GetGameRepresentation
from pyxdameraulevenshtein import normalized_damerau_levenshtein_distance

def get_ents_with_concepts(item, concept_order):
    re_obj = RankEntities()
    ts = re_obj.get_ranked_teams(item)
    ps = re_obj.get_ranked_players(item)
    t_combs = re_obj.get_ranked_teams_comb(item)
    pt_combs = re_obj.get_ranked_player_team_comb(item)
    pp_combs = re_obj.get_ranked_players_comb(item)
    ts_idx, ps_idx, t_combs_idx, pt_combs_idx, pp_combs_idx = 0, 0, 0, 0, 0
    concepts = concept_order
    delim = '|'
    new_concepts = []
    for c in concepts:
        concept_type = c.split('-')[0]
        if concept_type == 'T':
            try:
                new_concepts.append(f"{ts[ts_idx]}{delim}{c}")
                ts_idx += 1
            except:
                new_concepts.append(f"{ts[0]}{delim}{c}")
        elif concept_type == 'P':
            new_concepts.append(f"{ps[ps_idx]}{delim}{c}")
            ps_idx += 1
        elif concept_type == 'T&T':
            try:
                new_concepts.append(f"{t_combs[t_combs_idx]}{delim}{c}")
                t_combs_idx += 1
            except:
                new_concepts.append(f"{t_combs[0]}{delim}{c}")
        elif concept_type == 'P&T':
            new_concepts.append(f"{pt_combs[pt_combs_idx]}{delim}{c}")
            pt_combs_idx += 1
        elif concept_type == 'P&P':
            new_concepts.append(f"{pp_combs[pp_combs_idx]}{delim}{c}")
            pp_combs_idx += 1
    return new_concepts

def get_sol_idx(dists, cb_sol, idx1):
    dists_1d = dists.ravel()
    dists_sorted = np.argsort(dists_1d)
    sol_idx = dists_sorted[0]
    for _, d in enumerate(dists_sorted[:10]):
        sol = cb_sol[d]
        if len(sol) > 10:
            # print(idx1, 'more than 10 concepts')
            sol_idx = d
            break
    return int(sol_idx)

def main():
    """
    Representation of game data.
    Let's say we have a game with two teams (T1 & T2), and each team has 3 players ((P1, P2, P3) from T1 and (P4, P5, P6) from T2). Teams and players have their own attributes/features (e.g. PTS, REB, AST, STL, BLK).
    Assuming that T2 won; and P1's PTS is higher than P2's PTS such that P1 > P2 > P3 in T1 and similarly P4 > P5 > P6 in T2, we can represent the game as:
    [T2, T1, P4, P5, P6, P1, P2, P3, T2P4, T2P5, T2P6, T1P1, T1P2, T1P3, P4P5, P4P6, P5P6, P1P2, P1P3, P2P3]
    [teams, players, teams|players, players|players]
    [win_team, lose_team, win_players, lose_players, win_team|win_players, lose_team|lose_players, win_players*win_players, lose_players*lose_players]
    """
    dataset = load_dataset('GEM/sportsett_basketball')
    part = 'test'
    # non_rg_obj = NonRGMetrics()
    game_rep_obj = GetGameRepresentation()
    test_set_sents_gold = json.load(open(f'data/{part}_data_ct.json'))
    cb = json.load(open('cbs/cb_stage1.json'))
    cb_prob = np.array([item['problem'] for item in tqdm(cb)])
    cb_sol = [item['solution'] for item in tqdm(cb)]
    print("cb_prob.shape, len(cb_sol)", cb_prob.shape, len(cb_sol))

    test_set_sol_pred = []
    test_set_sol_gold = []

    for idx, entry in tqdm(enumerate(dataset[f'{part}'])):
        # if idx < 2:
        target_problem_rep = np.array(game_rep_obj.get_full_game_repr(entry))
        dists = euclidean_distances(cb_prob, target_problem_rep.reshape(1, -1))
        sol_idx = get_sol_idx(dists, cb_sol, idx)
        # dists_1d = dists.ravel()
        # sol_idx = dists_1d.argmin()
        entry_sol = cb_sol[sol_idx] # this is the ordered concept list. now we need entities for each concept in the list.
        enrty_sol_ents_with_concepts = get_ents_with_concepts(entry, entry_sol)
        test_set_sol_pred.append(enrty_sol_ents_with_concepts)

        # gold_sents = list(filter(lambda x: x['gem_id'] == entry['gem_id'], test_set_sents_gold))
        # eco_obj = ExtractConceptOrder()
        # test_set_sol_gold.append(eco_obj.extract_concept_order(entry, gold_sents))

    # non_rg_co_score = non_rg_obj.calc_dld(test_set_sol_gold, test_set_sol_pred)
    # print(f'New Sys Non-RG CO-Score: {non_rg_co_score}')

    # all_scores = [1 - normalized_damerau_levenshtein_distance(orig, pred) for orig, pred in zip(test_set_sol_gold, test_set_sol_pred)]
    # cbr_co_score = np.mean(all_scores)
    # print(f'New Sys CO score: {cbr_co_score}')

    # with open(f'sportsett/output/co_scores.txt', 'a') as f:
    #     f.write(f'New Sys Non-RG CO-Score: {non_rg_co_score}\n\n\n')

    json.dump(test_set_sol_pred, open(f'sportsett/output/new_sys/concepts1.json', 'w'), indent='\t')
    # json.dump(test_set_sol_gold, open(f'sportsett/output/new_sys/gold.json', 'w'), indent='\t')


if __name__ == '__main__':
    main()