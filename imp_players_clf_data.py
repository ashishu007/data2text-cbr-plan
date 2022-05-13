import time, json, argparse
import numpy as np
from tqdm import tqdm
from utils import ExtractEntities, GetEntityRepresentation
from datasets import load_dataset
np.random.seed(0)

argparser = argparse.ArgumentParser()
argparser.add_argument('-season', '--season', type=str, default='2014', \
                        choices=['2014', '2015', '2016', '2017', '2018'])
args = argparser.parse_args()
season = args.season
print(args, season)

# season = "2014"
if season in ['2014', '2015', '2016']:
    part = 'train'
if season in ['2017']:
    part = 'validation'
if season in ['2018']:
    part = 'test'
data_split = json.load(open(f'data/seasonal_splits.json'))
train_ids = data_split[f'{season}']['train']
test_ids = data_split[f'{season}']['test']
validation_ids = data_split[f'{season}']['validation']

dataset = load_dataset('GEM/sportsett_basketball')
ee_obj = ExtractEntities()
ger_obj = GetEntityRepresentation()
part_sents = json.load(open(f'data/{part}_data_ct.json'))

for gem_ids in [train_ids, validation_ids]:
    X, y = [], []
    for idx_sc, score_dict in tqdm(enumerate(dataset[f'{part}'])):
        if score_dict['gem_id'] not in gem_ids:
            continue
        time_s = time.time()
        hbs = score_dict['teams']['home']['box_score']
        vbs = score_dict['teams']['vis']['box_score']
        hpts = int(score_dict['teams']['home']['line_score']['game']['PTS'])
        vpts = int(score_dict['teams']['vis']['line_score']['game']['PTS'])
        win = 'HOME' if hpts > vpts else 'VIS'
        gem_id = score_dict['gem_id']

        time_coref_s = time.time()
        idx_sents = list(filter(lambda x: x['gem_id'] == gem_id, part_sents))
        summary_sentences = [x['coref_sent'] for x in idx_sents]
        time_coref_e = time.time()

        hftrs, hlbs = [], []
        vftrs, vlbs = [], []
        players_mentioned = []
        time_ext_ents_s = time.time()
        for sent in list(summary_sentences)[1:]:
            all_ents, teams, players = ee_obj.get_all_ents(score_dict)
            player_ents_unresolved = ee_obj.extract_entities(players, sent)
            team_ents_unresolved = ee_obj.extract_entities(teams, sent)
            player_ents = ee_obj.get_full_player_ents(player_ents_unresolved, score_dict)
            team_ents = ee_obj.get_full_team_ents(team_ents_unresolved, score_dict)
            players_mentioned.extend(player_ents)
        players_mentioned = list(set(players_mentioned))
        time_ext_ents_e = time.time()

        time_sort_ents_s = time.time()
        home_sorted_idx = ger_obj.sort_players_by_pts(score_dict, type='HOME')
        vis_sorted_idx = ger_obj.sort_players_by_pts(score_dict, type='VIS')
        time_sort_ents_e = time.time()

        time_h_ftrs_s = time.time()
        for player_idx in home_sorted_idx:
            winner = 1 if win == 'HOME' else 0
            player = hbs[player_idx]
            if player['name'] in players_mentioned:
                hftrs.append(ger_obj.get_one_player_data(player, winner=winner))
                hlbs.append(1)
            else:
                hftrs.append(ger_obj.get_one_player_data(player, winner=winner))
                hlbs.append(0)
        htemp_ftrs = [ger_obj.get_empty_bs_dict(winner=winner) for _ in range(ger_obj.NUM_PLAYERS - len(home_sorted_idx))]
        htemp_lbs = [0 for _ in range(ger_obj.NUM_PLAYERS - len(home_sorted_idx))]
        if len(htemp_ftrs) > 0:
            hftrs.extend(htemp_ftrs)
            hlbs.extend(htemp_lbs)
        time_h_ftrs_e = time.time()

        time_v_ftrs_s = time.time()
        for player_idx in vis_sorted_idx:
            winner = 1 if win == 'VIS' else 0
            player = vbs[player_idx]
            if player['name'] in players_mentioned:
                vftrs.append(ger_obj.get_one_player_data(player, winner=winner))
                vlbs.append(1)
            else:
                vftrs.append(ger_obj.get_one_player_data(player, winner=winner))
                vlbs.append(0)
        vtemp_ftrs = [ger_obj.get_empty_bs_dict(winner=winner) for _ in range(ger_obj.NUM_PLAYERS - len(vis_sorted_idx))]
        vtemp_lbs = [0 for _ in range(ger_obj.NUM_PLAYERS - len(vis_sorted_idx))]
        if len(vtemp_ftrs) > 0:
            vftrs.extend(vtemp_ftrs)
            vlbs.extend(vtemp_lbs)
        time_v_ftrs_e = time.time()

        time_ftrs_s = time.time()
        ftrs = hftrs[:ger_obj.NUM_PLAYERS] + vftrs[:ger_obj.NUM_PLAYERS]
        lbs = hlbs[:ger_obj.NUM_PLAYERS] + vlbs[:ger_obj.NUM_PLAYERS]
        ftrs_arr = np.array([list(i.values()) for i in ftrs])
        time_ftrs_e = time.time()

        clf_ftrs = []
        time_ftrs_1d_s = time.time()
        for idx, item in enumerate(ftrs_arr):
            clf_ftr = item.ravel()
            temp = np.delete(ftrs_arr, idx, axis=0).ravel()
            clf_ftr = np.append(clf_ftr, temp)
            clf_ftrs.append(clf_ftr)
        time_ftrs_1d_e = time.time()

        clf_ftrs = np.array(clf_ftrs)
        lbs_arr = np.array(lbs)

        X.extend(clf_ftrs)
        y.extend(lbs_arr)

        time_e = time.time()

    out_dir = f'player_clf/data/{season}'

    X1 = np.array(X)
    y1 = np.array(y)
    print(X1.shape, y1.shape)

    if len(gem_ids) == 830:
        print('Training...')
        indices = np.random.permutation(len(X1))
        X_train = X1[indices[:int(len(X1)*0.8)]]
        y_train = y1[indices[:int(len(X1)*0.8)]]
        X_validation = X1[indices[int(len(X1)*0.8):]]
        y_validation = y1[indices[int(len(X1)*0.8):]]
        np.savez(f'{out_dir}/X_train.npz', X_train)
        np.savez(f'{out_dir}/y_train.npz', y_train)
        np.savez(f'{out_dir}/X_validation.npz', X_validation)
        np.savez(f'{out_dir}/y_validation.npz', y_validation)
    elif len(gem_ids) == 200:
        print('Validating...')
        np.savez(f'{out_dir}/X_test.npz', X1)
        np.savez(f'{out_dir}/y_test.npz', y1)
