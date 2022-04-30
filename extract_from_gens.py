"""
Evaluate Generated Summaries with on Concept-Ordering score
"""

import argparse, json
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from utils import ExtractConceptOrder
# from non_rg import NonRGMetrics
from clf_utils import ContentTypeData, MultiLabelClassifier
from coref_resolve import CorefResolver
from pyxdameraulevenshtein import normalized_damerau_levenshtein_distance

def get_ct_list_from_arr(arr):
    ct_list = []
    if arr[0] == 1:
        ct_list.append('B')
    if arr[1] == 1:
        ct_list.append('W')
    if arr[2] == 1:
        ct_list.append('A')
    return ct_list

def main(SYS_NAME='ent'):
    print(f'Constructing...')

    ctd = ContentTypeData()
    clf = MultiLabelClassifier()
    coref_obj = CorefResolver()
    eco_obj = ExtractConceptOrder()
    # non_rg_obj = NonRGMetrics()

    dataset = load_dataset('GEM/sportsett_basketball')
    # test_set_sents_gold = json.load(open(f'data/test_data_ct.json'))
    gen_out = open(f'gens/{SYS_NAME}.txt', 'r').readlines()
    gen_out = [line.strip() for line in gen_out]

    print(f'Constructed!!!')

    # test_set_sol_gold = []
    gen_concept_order = []
    for idx, gen in tqdm(enumerate(gen_out)):
        if idx % 100 == 0:
            print(f'\n\nthis is {idx}\n\n')
        entry = dataset[f'test'][idx]

        all_sents = coref_obj.process_one_summary(gen)
        ner_abs_sents = ctd.abstract_sents(all_sents)
        ct_y = clf.predict_multilabel_classif(ner_abs_sents)
        all_sents_with_ct = [{"coref_sent": sent, "content_types": get_ct_list_from_arr(ct_y[idx])} for idx, sent in enumerate(all_sents)]

        gen_concept_order.append(eco_obj.extract_concept_order(entry, all_sents_with_ct))

        # gold_sents = list(filter(lambda x: x['gem_id'] == entry['gem_id'], test_set_sents_gold))
        # test_set_sol_gold.append(eco_obj.extract_concept_order(entry, gold_sents))

    # non_rg_co_score = non_rg_obj.calc_dld(test_set_sol_gold, gen_concept_order)
    # print(f'{SYS_NAME.upper()} Non-RG CO-Score: {non_rg_co_score}')

    # all_scores = [1 - normalized_damerau_levenshtein_distance(orig, pred) for orig, pred in zip(test_set_sol_gold, gen_concept_order)]
    # gen_co_score = np.mean(all_scores)
    # print(f'{SYS_NAME.upper()} CO-Score: {gen_co_score}')

    # with open(f'sportsett/output/co_scores.txt', 'a') as f:
    #     f.write(f'{SYS_NAME.upper()} CO-Score: {gen_co_score}\n\n\n')
    # json.dump({f'{SYS_NAME}': non_rg_co_score}, open(f'sportsett/output/{SYS_NAME}/score.json', 'w'), indent='\t')

    json.dump(gen_concept_order, open(f'sportsett/output/{SYS_NAME}/concepts.json', 'w'), indent='\t')
    # json.dump(test_set_sol_gold, open(f'sportsett/output/{SYS_NAME}/gold.json', 'w'), indent='\t')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sys', '-sys', type=str, default='ent', choices=['ent', 'hir', 'mp', 'cbr', 'temp'])
    args = parser.parse_args()
    main(args.sys)
    print('Done!')
