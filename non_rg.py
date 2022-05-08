"""
# usage:
    python non_rg_metrics.py gold_tuple_fi pred_tuple_fi
"""
import json, sys
import pandas as pd
from pyxdameraulevenshtein import normalized_damerau_levenshtein_distance

class NonRGMetrics:

    full_names = ['Atlanta Hawks', 'Boston Celtics', 'Brooklyn Nets',
                'Charlotte Hornets', 'Chicago Bulls', 'Cleveland Cavaliers',
                'Detroit Pistons', 'Indiana Pacers', 'Miami Heat',
                'Milwaukee Bucks', 'New York Knicks', 'Orlando Magic',
                'Philadelphia 76ers', 'Toronto Raptors', 'Washington Wizards',
                'Dallas Mavericks', 'Denver Nuggets', 'Golden State Warriors',
                'Houston Rockets', 'Los Angeles Clippers', 'Los Angeles Lakers',
                'Memphis Grizzlies', 'Minnesota Timberwolves',
                'New Orleans Pelicans', 'Oklahoma City Thunder', 'Phoenix Suns',
                'Portland Trail Blazers', 'Sacramento Kings', 'San Antonio Spurs',
                'Utah Jazz']

    cities, teams = set(), set()
    ec = {}  # equivalence classes
    for team in full_names:
        pieces = team.split()
        if len(pieces) == 2:
            ec[team] = [pieces[0], pieces[1]]
            cities.add(pieces[0])
            teams.add(pieces[1])
        elif pieces[0] == "Portland":  # only 2-word team
            ec[team] = [pieces[0], " ".join(pieces[1:])]
            cities.add(pieces[0])
            teams.add(" ".join(pieces[1:]))
        else:  # must be a 2-word City
            ec[team] = [" ".join(pieces[:2]), pieces[2]]
            cities.add(" ".join(pieces[:2]))
            teams.add(pieces[2])


    def same_ent(self, e1, e2):
        if e1 in self.cities or e1 in self.teams:
            return e1 == e2 or any((e1 in fullname and e2 in fullname for fullname in self.full_names))
        else:
            return e1 in e2 or e2 in e1


    def trip_match(self, t1, t2, ent_or_concept='concepts'):
        # return t1[1] == t2[1] and t1[2] == t2[2] and self.same_ent(t1[0], t2[0])
        # return t1[1] == t2[1] and self.same_ent(t1[0], t2[0])
        if ent_or_concept == 'concepts':
            return t1[1] == t2[1]
        elif ent_or_concept == 'entities':
            return self.same_ent(t1[0], t2[0])


    def dedup_triples(self, triplist):
        """
        this will be inefficient but who cares
        """
        dups = set()
        for i in range(1, len(triplist)):
            for j in range(i):
                if self.trip_match(triplist[i], triplist[j]):
                    dups.add(i)
                    break
        return [thing for i, thing in enumerate(triplist) if i not in dups]


    def get_triples(self, fi):
        all_triples = []
        curr = []
        with open(fi) as f:
            for line in f:
                if line.isspace():
                    all_triples.append(self.dedup_triples(curr))
                    curr = []
                else:
                    pieces = line.strip().split('|')
                    curr.append(tuple(pieces))
        if len(curr) > 0:
            all_triples.append(self.dedup_triples(curr))
        return all_triples


    def get_triples_new(self, fi, ent_or_concept='concepts'):
        delim = "|"
        js = json.load(open(f'{fi}'))
        # data = [[tuple(val.split(delim)) for val in item] for item in js]
        data = []
        for item in js:
            temp = []
            for val in item:
                if ent_or_concept == 'concepts':
                    temp.append(tuple(['', val.split(delim)[1]])) # only using the concepts
                # temp.append(tuple([val.split(delim)[0], val.split(delim)[1]])) # only using the concepts
                elif ent_or_concept == 'entities':
                    ents = val.split(delim)[0].split(' & ')
                    for ent in ents:
                        # temp.append(tuple([ent, val.split(delim)[1]]))
                        temp.append(tuple([ent, '']))
            data.append(temp)
        return data


    def calc_precrec(self, goldfi, predfi, ent_or_concept='concepts'):
        # gold_triples = goldfi
        # pred_triples = predfi
        # gold_triples = self.get_triples(goldfi)
        # pred_triples = self.get_triples(predfi)

        gold_triples = self.get_triples_new(goldfi, ent_or_concept=ent_or_concept)
        pred_triples = self.get_triples_new(predfi, ent_or_concept=ent_or_concept)

        total_tp, total_predicted, total_gold = 0, 0, 0
        # print(len(gold_triples), len(pred_triples))
        # print(gold_triples[0], pred_triples[0])
        # print(gold_triples[-1], pred_triples[-1])
        # print(set([len(item) for item1 in gold_triples for item in item1]))
        # print(set([len(item) for item1 in pred_triples for item in item1]))
        assert len(gold_triples) == len(pred_triples)

        for i, triplist in enumerate(pred_triples):
            tp = sum((1 for j in range(len(triplist))
                    if any(self.trip_match(triplist[j], gold_triples[i][k], ent_or_concept=ent_or_concept)
                            for k in range(len(gold_triples[i])))))
            total_tp += tp
            total_predicted += len(triplist)
            total_gold += len(gold_triples[i])
        avg_prec = float(total_tp) / total_predicted
        avg_rec = float(total_tp) / total_gold
        avg_jac = float(total_tp) / (total_predicted + (total_gold - total_tp))
        avg_f1 = 2 * float(avg_prec * avg_rec) / (avg_prec + avg_rec)
        avg_dice = float(2 * total_tp) / (total_predicted + total_gold)
        avg_f2 = 5 * float(avg_prec * avg_rec) / ((4 * avg_prec) + avg_rec)
        # print("totals:", total_tp, total_predicted, total_gold)
        # print("prec:", avg_prec, "rec:", avg_rec)
        return avg_prec, avg_rec, avg_jac, avg_f1, avg_dice, avg_f2


    def norm_dld(self, l1, l2, ent_or_concept='concepts'):
        ascii_start = 0
        # make a string for l1
        # all triples are unique...
        s1 = ''.join((chr(ascii_start + i) for i in range(len(l1))))
        s2 = ''
        next_char = ascii_start + len(s1)
        for j in range(len(l2)):
            found = None
            # next_char = chr(ascii_start+len(s1)+j)
            for k in range(len(l1)):
                if self.trip_match(l2[j], l1[k], ent_or_concept=ent_or_concept):
                    found = s1[k]
                    # next_char = s1[k]
                    break
            if found is None:
                s2 += chr(next_char)
                next_char += 1
                assert next_char <= 128
            else:
                s2 += found
        # return 1- , since this thing gives 0 to perfect matches etc
        return 1.0 - normalized_damerau_levenshtein_distance(s1, s2)


    def calc_dld(self, goldfi, predfi, ent_or_concept='concepts'):
        # gold_triples = goldfi
        # pred_triples = predfi
        # gold_triples = self.get_triples(goldfi)
        # pred_triples = self.get_triples(predfi)
        gold_triples = self.get_triples_new(goldfi, ent_or_concept=ent_or_concept)
        pred_triples = self.get_triples_new(predfi, ent_or_concept=ent_or_concept)
        assert len(gold_triples) == len(pred_triples)
        total_score = 0
        for i, triplist in enumerate(pred_triples):
            total_score += self.norm_dld(triplist, gold_triples[i], ent_or_concept=ent_or_concept)
        avg_score = float(total_score) / len(pred_triples)
        # print("avg score:", avg_score)
        return avg_score


obj = NonRGMetrics()
ent_or_concept = sys.argv[1]
systems = ['first', 'mean', 'median', 'new_sys', 'cbr', 'temp', 'mp', 'ent', 'hir', 'imp_player_new_sys', 'imp_player_median']#, 'max']
if ent_or_concept != 'len':
    res = {}
    print(f'\nThis is {ent_or_concept}\n')
    for sys_name in systems:
        sys_res = {}
        file_name = 'concepts' if sys_name == 'new_sys' else 'concepts'
        predfi = f'sportsett/output/{sys_name}/{file_name}.json'
        # print(predfi)
        goldfi = f'sportsett/output/new_sys/gold.json'
        prec, rec, jac, f1, dice, f2 = obj.calc_precrec(goldfi, predfi, ent_or_concept=ent_or_concept)
        dld = obj.calc_dld(goldfi, predfi)
        score_str = f'\n{sys_name.upper()}\t\t||\tPrec: {prec*100:.2f}\tRec: {rec*100:.2f}\tF1: {f1*100:.2f}\tJac: {jac*100:.2f}\tF2: {f2*100:.2f}\n'
        sys_res['f2'] = f"{f2*100:.2f}"
        sys_res['prec'] = f"{prec*100:.2f}"
        sys_res['rec'] = f"{rec*100:.2f}"
        sys_res['f1'] = f"{f1*100:.2f}"
        sys_res['jac'] = f"{jac*100:.2f}"
        sys_res['dld'] = f"{dld*100:.2f}"
        res[sys_name] = sys_res
        # print(f'\n{sys_name.upper()}\t||\tPrec: {prec*100:.2f}\tRec: {rec*100:.2f}\tDLD: {dld*100:.2f}\tJac: {jac*100:.2f}\tF1: {f1*100:.2f}\tDICE: {dice*100:.2f}\tF2: {f2*100:.2f}\n')
        # print(f'\n{sys_name.upper()}\t||\tPrec: {prec*100:.2f}\tRec: {rec*100:.2f}\tF1: {f1*100:.2f}\tJac: {jac*100:.2f}\tF2: {f2*100:.2f}\n')
        # print(score_str)
    # open(f'sportsett/output/results_{ent_or_concept}.txt', 'w').write(score_str)
    # json.dump(res, open(f'sportsett/output/results_{ent_or_concept}.json', 'w'), indent='\t')
    df = pd.DataFrame(res)
    df.transpose().to_csv(f'sportsett/output/results_{ent_or_concept}.csv')

else:
    # concept length
    systems.append('gold')
    lens = {}
    for sys_name in systems:
        if sys_name == 'gold':
            tpls = obj.get_triples_new(f'sportsett/output/new_sys/gold.json')
        else:
            tpls = obj.get_triples_new(f'sportsett/output/{sys_name}/concepts.json')
        avg = 0
        for tpl in tpls:
            avg += len(tpl)
        avg /= len(tpls)
        print(f"{sys_name}\t{avg}")
        lens[sys_name] = avg
    json.dump(lens, open(f'sportsett/output/concept_lengths.json', 'w'), indent='\t')
