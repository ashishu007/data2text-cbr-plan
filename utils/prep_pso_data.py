import json
from tqdm import tqdm
from datasets import load_dataset
from utils import ExtractEntities, GetEntityRepresentation

part = f'train'
js = json.load(open(f'data/{part}_data_ct.json'))
ee_obj = ExtractEntities()
gep_obj = GetEntityRepresentation(popularity=True, season='all')
dataset = load_dataset('GEM/sportsett_basketball')

print(f'Creating Player Mention Data')
player_mention_data = []
for idx, item in tqdm(enumerate(dataset[f'{part}'])):
    player_in_game_reps, player_in_game_labs = [], []
    home_player_mention_idxs, vis_player_mention_idxs = [], []

    home_full_names = [player['name'] for player in item['teams']['home']['box_score']]
    home_first_names = [player['first_name'] for player in item['teams']['home']['box_score']]
    home_last_names = [player['last_name'] for player in item['teams']['home']['box_score']]

    vis_full_names = [player['name'] for player in item['teams']['vis']['box_score']]
    vis_first_names = [player['first_name'] for player in item['teams']['vis']['box_score']]
    vis_last_names = [player['last_name'] for player in item['teams']['vis']['box_score']]

    all_ents, team_ents, player_ents = ee_obj.get_all_ents(item)
    all_sents = list(filter(lambda x: x['gem_id'] == item['gem_id'], js))    
    unique_summary_idx = set([x['summary_idx'] for x in all_sents])

    for us_id in unique_summary_idx:
        summary_sents = list(filter(lambda x: x['summary_idx'] == us_id, all_sents))
        for sent in summary_sents:
            player_ents_in_sent = ee_obj.extract_entities(player_ents, sent['coref_sent'])

            if len(player_ents_in_sent) == 1:
                home_player_idx = -1
                vis_player_idx = -1
                if player_ents_in_sent[0] in home_full_names:
                    for idx2, player in enumerate(item['teams']['home']['box_score']):
                        if player['name'] == player_ents_in_sent[0]:
                            home_player_idx = idx2
                elif player_ents_in_sent[0] in home_first_names:
                    for idx2, player in enumerate(item['teams']['home']['box_score']):
                        if player['first_name'] == player_ents_in_sent[0]:
                            home_player_idx = idx2
                elif player_ents_in_sent[0] in home_last_names:
                    for idx2, player in enumerate(item['teams']['home']['box_score']):
                        if player['last_name'] == player_ents_in_sent[0]:
                            home_player_idx = idx2
                
                if player_ents_in_sent[0] in vis_full_names:
                    for idx2, player in enumerate(item['teams']['vis']['box_score']):
                        if player['name'] == player_ents_in_sent[0]:
                            vis_player_idx = idx2
                elif player_ents_in_sent[0] in vis_first_names:
                    for idx2, player in enumerate(item['teams']['vis']['box_score']):
                        if player['first_name'] == player_ents_in_sent[0]:
                            vis_player_idx = idx2
                elif player_ents_in_sent[0] in vis_last_names:
                    for idx2, player in enumerate(item['teams']['vis']['box_score']):
                        if player['last_name'] == player_ents_in_sent[0]:
                            vis_player_idx = idx2

                if home_player_idx != -1:
                    home_player_mention_idxs.append(home_player_idx)
                if vis_player_idx != -1:
                    vis_player_mention_idxs.append(vis_player_idx)

        home_team_points = item['teams']['home']['line_score']['game']['PTS']
        vis_team_points = item['teams']['vis']['line_score']['game']['PTS']
        home_bs = item['teams']['home']['box_score']
        vis_bs = item['teams']['vis']['box_score']
        home_sorted_player_ids = gep_obj.sort_players_by_pts(item, 'HOME')
        vis_sorted_player_ids = gep_obj.sort_players_by_pts(item, 'VIS')

        for player_id in home_sorted_player_ids:
            winner = 1 if home_team_points > vis_team_points else 0
            idx1_rep = list(gep_obj.get_one_player_data(home_bs[player_id], winner=winner).values())
            lab = 1 if player_id in home_player_mention_idxs else 0
            player_in_game_reps.append(idx1_rep)
            player_in_game_labs.append(lab)

        for player_id in vis_sorted_player_ids:
            winner = 1 if home_team_points < vis_team_points else 0
            idx1_rep = list(gep_obj.get_one_player_data(vis_bs[player_id], winner=winner).values())
            lab = 1 if player_id in vis_player_mention_idxs else 0
            player_in_game_reps.append(idx1_rep)
            player_in_game_labs.append(lab)

        player_mention_data.append({
            'features': player_in_game_reps,
            'labels': player_in_game_labs,
            "gem_id": item['gem_id'],
            "summary_idx": us_id
        })

print(f'Player mention data created, saving data for PSO training')
data_players = []
for idx, item in tqdm(enumerate(player_mention_data)):
    pos_exs = [item['features'][idx1] for idx1, item1 in enumerate(item['labels']) if item1 == 1]
    neg_exs = [item['features'][idx1] for idx1, item1 in enumerate(item['labels']) if item1 == 0]
    for pos, neg in zip(pos_exs, neg_exs[:len(pos_exs)]):
        ftr_rep = [pos[i] - neg[i] for i in range(len(pos))]
        ftr_rep[-1] = (ftr_rep[-1])/10
        data_players.append({"features": ftr_rep, "label": 1})
    
    for neg, pos in zip(neg_exs[:len(pos_exs)], pos_exs):
        ftr_rep = [neg[i] - pos[i] for i in range(len(pos))]
        ftr_rep[-1] = (ftr_rep[-1])/10
        data_players.append({"features": ftr_rep, "label": 0})

json.dump(data_players, open(f'ranking_outs/players4pso_{part}_pop.json', 'w'))
print(f'Player mention data saved for PSO training')

# data_teams = []
# for idx, item in tqdm(enumerate(dataset[f'{part}'])):
#     hls = item['teams']['home']['line_score']['game']
#     vls = item['teams']['vis']['line_score']['game']
#     winner = 'HOME' if int(hls['PTS']) > int(vls['PTS']) else 'VIS'
#     hrep = gep_obj.get_team_line(item, type='HOME', winner=winner)
#     vrep = gep_obj.get_team_line(item, type='VIS', winner=winner)
#     hwin = 1 if winner == 'HOME' else 0
#     vwin = 1 if winner == 'VIS' else 0
#     ftrs1 = [hrep[key] - vrep[key] for key, _ in hrep.items()]
#     data_teams.append({'features': ftrs1, 'label': hwin})
#     ftrs2 = [vrep[key] - hrep[key] for key, _ in vrep.items()]
#     data_teams.append({'features': ftrs2, 'label': vwin})

# json.dump(data_teams, open(f'ranking_outs/teams4pso_{part}.json', 'w'))
