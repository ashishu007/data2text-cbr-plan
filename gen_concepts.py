import json, pickle, argparse
import numpy as np
from tqdm import tqdm
from utils.utils import GetEntityRepresentation, GetGameRepresentation
from utils.entity_ranking import RankEntities
from datasets import load_dataset
from sklearn.metrics.pairwise import euclidean_distances, cosine_distances
from utils.imp_players_utility import get_imp_players_test, get_game_repr_imp_players
from sentence_transformers import SentenceTransformer

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

def get_sol_idx(dists, cb_sol, top_k=15):
    dists_1d = dists.ravel()
    dists_sorted = np.argsort(dists_1d)
    sol_idx = dists_sorted[0]
    max_len = 0
    for _, d in enumerate(dists_sorted[:top_k]):
        sol = cb_sol[d]
        if len(sol) > 10:
            sol_idx = d
            break
    return cb_sol[int(sol_idx)]

def get_sol_by_median(dists, cb_sol, top_k=15):
    dists_1d = dists.ravel()
    dists_sorted = np.argsort(dists_1d)
    sols_top_k_idx = dists_sorted[:top_k]
    sols_top_k = [cb_sol[sol_idx] for sol_idx in sols_top_k_idx]
    sols_len = [len(i) for i in sols_top_k]
    median_sol_len_idx = np.argsort(sols_len)[int(len(sols_len)//2)]
    return sols_top_k[median_sol_len_idx]

def get_solution(dists, cb_sol, reuse_type='long', ret_set_size=15):
    if reuse_type == 'long':
        entry_sol = get_sol_idx(dists, cb_sol)
    elif reuse_type == 'median':
        entry_sol =  get_sol_by_median(dists, cb_sol)
    elif reuse_type == 'first':
        dists_1d = dists.ravel()
        dists_sorted = np.argsort(dists_1d)
        sol_idx = dists_sorted[0]
        entry_sol = cb_sol[sol_idx]
    return entry_sol

def main(POP=False, WEIGHTED=False, SEASON='2014', TOPK=15):

    bstr = f"*"*100
    print(f"\nInside main()\n{bstr}")
    print(f"Popularity: {POP}\tWeighing: {WEIGHTED}\tSeason: {SEASON}\tRetrieval Set Size: {TOPK}\n{bstr}\n")

    season = SEASON

    if season in ['2014', '2015', '2016', 'bens']:
        part = 'train'
    if season in ['2017', 'juans']:
        part = 'validation'
    if season in ['2018', 'all']:
        part = 'test'
    data_split = json.load(open(f'data/seasonal_splits.json'))
    train_ids = data_split[f'{season}']['train']
    test_ids = data_split[f'{season}']['test']
    validation_ids = data_split[f'{season}']['validation']

    dataset = load_dataset('GEM/sportsett_basketball')
    embedding_model = SentenceTransformer('bert-base-nli-mean-tokens')
    imp_player_clf = pickle.load(open(f'player_clf/model/model_{season}.pkl', 'rb'))
    ger_obj = GetEntityRepresentation(popularity=POP)
    ggr_obj = GetGameRepresentation()
    re_obj = RankEntities()
    # part = 'test'

    players = 'imp' # ['all', 'imp']
    ftrs = 'num' # ['num', 'set', 'text']
    dist_type = 'cosine' # ['euclidean', 'cosine']
    reuse = 'long' # ['long', 'median', 'first']
    cb_sol = json.load(open(f'cbs/{season}/cb_sol.json'))
    player_weights = np.load(f'ranking_outs/players_weights_pop.npy')

    if players == 'imp':
        all_ftrs_types = ['set']#, 'text']#, 'text', 'set']
    elif players == 'all':
        all_ftrs_types = ['num']

    for ftrs in all_ftrs_types:
        for dist_type in ['cosine']:#, 'euclidean']:#, 'cosine']:
            for reuse in ['long', 'median']:#, 'first']:

                print(f"\nPlayers: {players}\tFeature Type: {ftrs}")
                prob_file = f"{players}_players_{ftrs}_ftrs_pop_cb_prob" if POP else f"{players}_players_{ftrs}_ftrs_cb_prob"
                cb_prob = np.load(f'cbs/{season}/{prob_file}.npz')['arr_0']
                print(f"Prob Set: {cb_prob.shape}\tSol Set: {len(cb_sol)}\n{prob_file}")
                print(f"\nPlayers: {players}\tFeature Type: {ftrs}\tDist Type: {dist_type}\tReuse Type: {reuse}")
                test_set_sol_pred = []

                for _, entry in tqdm(enumerate(dataset[f'{part}'])):
                    if entry['gem_id'] not in test_ids:
                        continue

                    if players == 'imp':
                        if POP:
                            ger_obj1 = GetEntityRepresentation()
                            imp_players = get_imp_players_test(entry, ger_obj1, imp_player_clf)
                        else:
                            imp_players = get_imp_players_test(entry, ger_obj, imp_player_clf)
                        target_problem_rep = get_game_repr_imp_players(entry, ger_obj, imp_players, embedding_model, ftrs_type=ftrs)
                    elif players == 'all':
                        assert ftrs == 'num'
                        target_problem_rep = ggr_obj.get_full_game_repr(entry)

                    if WEIGHTED:
                        target_problem_rep = target_problem_rep * player_weights
                        cb_prob = cb_prob * player_weights

                    if dist_type == 'cosine':
                        dists = cosine_distances(cb_prob, target_problem_rep.reshape(1, -1))
                        # dists = cosine_distances(cb_prob*player_weights, target_problem_rep*player_weights.reshape(1, -1))
                    elif dist_type == 'euclidean':
                        dists = euclidean_distances(cb_prob, target_problem_rep.reshape(1, -1))
                        # dists = euclidean_distances(cb_prob*player_weights, target_problem_rep*player_weights.reshape(1, -1))

                    entry_sol = get_solution(dists, cb_sol, reuse_type=reuse, ret_set_size=TOPK)

                    enrty_sol_ents_with_concepts = get_ents_with_concepts(entry, entry_sol, re_obj)
                    test_set_sol_pred.append(enrty_sol_ents_with_concepts)
                
                sol_file = f"{players}_players-{ftrs}_ftrs-{dist_type}_sim-{reuse}_reuse-pop" if POP else f"{players}_players-{ftrs}_ftrs-{dist_type}_sim-{reuse}_reuse"
                sol_file = f"{sol_file}-weighted" if WEIGHTED else sol_file
                sol_file = f"{sol_file}-topk_{TOPK}" if TOPK != 15 else sol_file
                print(sol_file)
                json.dump(test_set_sol_pred, open(f'sportsett/concepts/{season}/{sol_file}.json', 'w'), indent='\t')

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--pop', '-pop', action='store_true')
    argparser.add_argument('--weighted', '-weighted', action='store_true')
    argparser.add_argument('-season', '--season', type=str, default='2014', \
                            choices=['2014', '2015', '2016', '2017', '2018', 'bens', 'all', 'juans'])
    argparser.add_argument('-topk', '--topk', type=str, default='15', \
                            choices=['5', '10', '15', '20', '25', '30'])
    args = argparser.parse_args()
    print(args)#, args.pop, type(args.pop), args.season, type(args.season))
    main(POP=args.pop, WEIGHTED=args.weighted, SEASON=args.season, TOPK=int(args.topk))
