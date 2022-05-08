import numpy as np

def get_imp_players_train(score_dict, summary_sentences, ee_obj):
    imp_players = []
    for sent in list(summary_sentences)[1:]:
        _, _, players = ee_obj.get_all_ents(score_dict)
        player_ents_unresolved = ee_obj.extract_entities(players, sent)
        player_ents = ee_obj.get_full_player_ents(player_ents_unresolved, score_dict)
        imp_players.extend(player_ents)
    return list(set(imp_players))

def get_imp_players_test(score_dict, ger_obj, clf_model):
    hbs = score_dict['teams']['home']['box_score']
    vbs = score_dict['teams']['vis']['box_score']
    hpts = int(score_dict['teams']['home']['line_score']['game']['PTS'])
    vpts = int(score_dict['teams']['vis']['line_score']['game']['PTS'])
    win = 'HOME' if hpts > vpts else 'VIS'
    hftrs, vftrs = [], []
    all_players = []

    home_sorted_idx = ger_obj.sort_players_by_pts(score_dict, type='HOME')
    vis_sorted_idx = ger_obj.sort_players_by_pts(score_dict, type='VIS')

    for player_idx in home_sorted_idx:
        winner = 1 if win == 'HOME' else 0
        player = hbs[player_idx]
        hftrs.append(ger_obj.get_one_player_data(player, winner=winner))
        all_players.append(player['name'])
    for _ in range(ger_obj.NUM_PLAYERS - len(home_sorted_idx)):
        hftrs.append(ger_obj.get_empty_bs_dict(winner=winner))
        all_players.append('None')

    for player_idx in vis_sorted_idx:
        winner = 1 if win == 'VIS' else 0
        player = vbs[player_idx]
        vftrs.append(ger_obj.get_one_player_data(player, winner=winner))
        all_players.append(player['name'])
    for _ in range(ger_obj.NUM_PLAYERS - len(vis_sorted_idx)):
        vftrs.append(ger_obj.get_empty_bs_dict(winner=winner))
        all_players.append('None')

    ftrs = hftrs[:ger_obj.NUM_PLAYERS] + vftrs[:ger_obj.NUM_PLAYERS]
    ftrs_arr = np.array([list(i.values()) for i in ftrs])

    clf_ftrs = []
    for idx1, item in enumerate(ftrs_arr):
        clf_ftr = item.ravel()
        temp = np.delete(ftrs_arr, idx1, axis=0).ravel()
        clf_ftr = np.append(clf_ftr, temp)
        clf_ftrs.append(clf_ftr)

    clf_ftrs = np.array(clf_ftrs)

    pred_y = clf_model.predict(clf_ftrs)
    imp_players = [all_players[i] for i in np.where(pred_y == 1)[0]]
    return imp_players

def get_game_repr_imp_players(score_dict, ger_obj, imp_players):
    hbs = score_dict['teams']['home']['box_score']
    vbs = score_dict['teams']['vis']['box_score']
    hpts = int(score_dict['teams']['home']['line_score']['game']['PTS'])
    vpts = int(score_dict['teams']['vis']['line_score']['game']['PTS'])
    win = 'HOME' if hpts > vpts else 'VIS'
    hftrs, vftrs = [], []

    home_sorted_idx = ger_obj.sort_players_by_pts(score_dict, type='HOME')
    vis_sorted_idx = ger_obj.sort_players_by_pts(score_dict, type='VIS')

    for player_idx in home_sorted_idx:
        winner = 1 if win == 'HOME' else 0
        player = hbs[player_idx]
        if player['name'] in imp_players:
            hftrs.append(ger_obj.get_one_player_data(player, winner=winner))

    for player_idx in vis_sorted_idx:
        winner = 1 if win == 'VIS' else 0
        player = vbs[player_idx]
        if player['name'] in imp_players:
            vftrs.append(ger_obj.get_one_player_data(player, winner=winner))

    ftrs = hftrs + vftrs if win == 'HOME' else vftrs + hftrs
    if len(ftrs) < 10:
        ftrs.extend([ger_obj.get_empty_bs_dict(winner=winner) for _ in range(10 - len(ftrs))])
    ftrs_arr = np.array([list(i.values()) for i in ftrs[:10]])

    return ftrs_arr.ravel()

