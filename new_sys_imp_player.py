import json, pickle
import numpy as np
from tqdm import tqdm
from utils import GetEntityRepresentation
from datasets import load_dataset
from entity_ranking import RankEntities
from sklearn.metrics.pairwise import euclidean_distances
from sel_imp_players import get_imp_players_test, get_game_repr_imp_players

def get_ents_with_concepts(item, concept_order, re_obj):
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

def get_sol_idx(dists, cb_sol):
    dists_1d = dists.ravel()
    dists_sorted = np.argsort(dists_1d)
    sol_idx = dists_sorted[0]
    max_len = 0
    for _, d in enumerate(dists_sorted[:15]):
        sol = cb_sol[d]
        # if len(sol) > max_len:
        #     max_len = len(sol)
        #     sol_idx = d
        if len(sol) > 10:
            # print(idx1, 'more than 10 concepts')
            sol_idx = d
            break
    return cb_sol[int(sol_idx)]

def get_sol_by_median(dists, cb_sol):
    dists_1d = dists.ravel()
    dists_sorted = np.argsort(dists_1d)
    sols_top_k_idx = dists_sorted[:15]
    sols_top_k = [cb_sol[sol_idx] for sol_idx in sols_top_k_idx]
    sols_len = [len(i) for i in sols_top_k]
    median_sol_len_idx = np.argsort(sols_len)[int(len(sols_len)//2)]
    return sols_top_k[median_sol_len_idx]


def get_sol_with_top_k(dists, cb_sol, mean_or_median='mean'):
    dists_1d = dists.ravel()
    dists_sorted = np.argsort(dists_1d)
    sols_top_k = dists_sorted[:15]
    all_sol_lens = []
    all_sol_concepts_dists = {}
    for sol_idx in sols_top_k:
        idx_sol = cb_sol[sol_idx]
        all_sol_lens.append(len(idx_sol))
        for c in idx_sol:
            if c not in all_sol_concepts_dists:
                all_sol_concepts_dists[c] = 1
            else:
                all_sol_concepts_dists[c] += 1
    all_sol_concepts_dists = sorted(all_sol_concepts_dists.items(), key=lambda kv: kv[1], reverse=True)
    all_sol_concepts_list = [c[0] for c in all_sol_concepts_dists]
    if mean_or_median == 'mean':
        new_sol_len = int(np.mean(all_sol_lens))
    elif mean_or_median == 'median':
        new_sol_len = int(np.median(all_sol_lens))
    elif mean_or_median == 'max':
        new_sol_len = int(np.max(all_sol_lens))
    new_sol_concepts = all_sol_concepts_list[:new_sol_len]
    return new_sol_concepts

def get_solution(dists, cb_sol, sys='new_sys'):
    if sys == 'new_sys':
        entry_sol = get_sol_idx(dists, cb_sol)
    elif sys == 'median':
        entry_sol =  get_sol_by_median(dists, cb_sol)
    elif sys == 'mean' or sys == 'max':
        entry_sol = get_sol_with_top_k(dists, cb_sol, mean_or_median=sys)
    elif sys == 'first':
        dists_1d = dists.ravel()
        dists_sorted = np.argsort(dists_1d)
        sol_idx = dists_sorted[0]
        entry_sol = cb_sol[sol_idx]
    return entry_sol

def main():
    """
    Representation of game data.
    Let's say we have a game with two teams (T1 & T2), and each team has 3 players ((P1, P2, P3) from T1 and (P4, P5, P6) from T2). Teams and players have their own attributes/features (e.g. PTS, REB, AST, STL, BLK).
    Assuming that T2 won; and P1's PTS is higher than P2's PTS such that P1 > P2 > P3 in T1 and similarly P4 > P5 > P6 in T2, we can represent the game as:
    [T2, T1, P4, P5, P6, P1, P2, P3, T2P4, T2P5, T2P6, T1P1, T1P2, T1P3, P4P5, P4P6, P5P6, P1P2, P1P3, P2P3]
    [teams, players, teams|players, players|players]
    [win_team, lose_team, win_players, lose_players, win_team|win_players, lose_team|lose_players, win_players*win_players, lose_players*lose_players]
    """

    # for sys in ['first', 'new_sys', 'mean', 'median']:#, 'max']:
    for sys in ['median']:
        print(f'\nthis is {sys}\n')
        dataset = load_dataset('GEM/sportsett_basketball')
        imp_player_clf = pickle.load(open('player_clf/model/model.pkl', 'rb'))
        part = 'test'
        ger_obj = GetEntityRepresentation()
        re_obj = RankEntities()
        cb_prob = np.load('cbs/imp_player_cb_prob.npz')['arr_0']
        cb_sol = json.load(open('cbs/imp_player_cb_sol.json'))
        print("cb_prob.shape, len(cb_sol)", cb_prob.shape, len(cb_sol))

        test_set_sol_pred = []
        for _, entry in tqdm(enumerate(dataset[f'{part}'])):
            imp_players = get_imp_players_test(entry, ger_obj, imp_player_clf)
            target_problem_rep = get_game_repr_imp_players(entry, ger_obj, imp_players)
            dists = euclidean_distances(cb_prob, target_problem_rep.reshape(1, -1))
            entry_sol = get_solution(dists, cb_sol, sys=sys)
            enrty_sol_ents_with_concepts = get_ents_with_concepts(entry, entry_sol, re_obj)
            test_set_sol_pred.append(enrty_sol_ents_with_concepts)

        json.dump(test_set_sol_pred, open(f'sportsett/output/imp_player_{sys}/concepts.json', 'w'), indent='\t')


if __name__ == '__main__':
    main()