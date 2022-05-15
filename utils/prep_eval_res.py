import json, pandas as pd, argparse

argparser = argparse.ArgumentParser()
argparser.add_argument('-season', '--season', type=str, default='2014', \
                        choices=['2014', '2015', '2016', '2017', '2018', 'bens', 'all', 'juans'])
args = argparser.parse_args()
season = args.season
print(season)

lens = json.load(open(f'sportsett/res/{season}/concept_lengths.json'))
concepts = pd.read_csv(f'sportsett/res/{season}/concepts.csv', index_col=0)
entities = pd.read_csv(f'sportsett/res/{season}/entities.csv', index_col=0)
concepts['systems'] = concepts.index
entities['systems'] = entities.index

all_systems = [row['systems'] for _, row in concepts.iterrows()]
new_systems = [row['systems'] for _, row in concepts.iterrows() if 'players' in row['systems']]
other_systems = [row['systems'] for _, row in concepts.iterrows() if 'players' not in row['systems']]
print(len(new_systems), len(other_systems), len(new_systems) + len(other_systems), len(all_systems))

dictionc = {'players': [], 'features': [], 'similarity': [], 'reuse': [], 'pop': [], 'f2': [], 'prec': [], 'rec': [], 'dld': []}
dictione = {'players': [], 'features': [], 'similarity': [], 'reuse': [], 'pop': [], 'f2': [], 'prec': [], 'rec': [], 'dld': []}

for sys in new_systems:
    info = [i.split('_') for i in sys.split('-')]
    players = info[0][0]
    ftrs = info[1][0]
    sim = info[2][0]
    reuse = info[3][0]
    if len(info) > 4:
        pop = True
    else:
        pop = False
    cscore = concepts.loc[concepts['systems'] == sys]
    escore = entities.loc[entities['systems'] == sys]
    
    dictionc['players'].append(players)
    dictionc['features'].append(ftrs)
    dictionc['similarity'].append(sim)
    dictionc['reuse'].append(reuse)
    dictionc['pop'].append(pop)
    dictionc['f2'].append(cscore['f2'].values[0])
    dictionc['prec'].append(cscore['prec'].values[0])
    dictionc['rec'].append(cscore['rec'].values[0])
    dictionc['dld'].append(cscore['dld'].values[0])
    
    dictione['players'].append(players)
    dictione['features'].append(ftrs)
    dictione['similarity'].append(sim)
    dictione['reuse'].append(reuse)
    dictione['pop'].append(pop)
    dictione['f2'].append(escore['f2'].values[0])
    dictione['prec'].append(escore['prec'].values[0])
    dictione['rec'].append(escore['rec'].values[0])
    dictione['dld'].append(escore['dld'].values[0])

dfe = pd.DataFrame(dictione)
dfe.to_csv(f'sportsett/res/{season}/eval_entities.csv', index=0)

dfc = pd.DataFrame(dictionc)
dfc.to_csv(f'sportsett/res/{season}/eval_concepts.csv', index=0)

if season == "all":
    dictionc = {'system': [], 'f2': [], 'prec': [], 'rec': [], 'dld': []}
    dictione = {'system': [], 'f2': [], 'prec': [], 'rec': [], 'dld': []}

    for sys in other_systems:
        cscore = concepts.loc[concepts['systems'] == sys]
        escore = entities.loc[entities['systems'] == sys]

        dictionc['system'].append(sys)
        dictionc['f2'].append(cscore['f2'].values[0])
        dictionc['prec'].append(cscore['prec'].values[0])
        dictionc['rec'].append(cscore['rec'].values[0])
        dictionc['dld'].append(cscore['dld'].values[0])
        
        dictione['system'].append(sys)
        dictione['f2'].append(escore['f2'].values[0])
        dictione['prec'].append(escore['prec'].values[0])
        dictione['rec'].append(escore['rec'].values[0])
        dictione['dld'].append(escore['dld'].values[0])

    dfc = pd.DataFrame(dictionc)
    dfc.to_csv(f'sportsett/res/{season}/eval_concepts_other.csv', index=0)
    dfe = pd.DataFrame(dictione)
    dfe.to_csv(f'sportsett/res/{season}/eval_entities_other.csv', index=0)
