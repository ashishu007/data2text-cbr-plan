# for ses in "2014" "2015" "2016" "bens" "juans" "all"
for ses in "all"
do
    echo "=========================================================="
    echo "Season: $ses"

    # echo "Creating Imp Player Classifier Data..."
    # python3 utils/imp_players_clf_data.py -season $ses

    # echo "Creating Imp Player Classifier..."
    # python3 player_clf/main.py -season $ses

    # echo "Creating Player Popularity Data..."
    # python3 utils/calc_popularity.py -season $ses

    # echo "Extracting Concepts from Gold Summaries..."
    # python3 utils/extract_from_gold.py -season $ses

    # echo "Creating Case-Base for ${ses} season..."
    # python3 create_cb.py -side both -season $ses -pop

    # for topk in 5 10
    # do
    #     echo "Generating Concepts on Test Set... with Retrieval Top-K: $topk"
    #     python3 gen_concepts.py -pop -season $ses -pop -weighted -topk $topk
    # done

    echo "Evaluating..."
    python3 utils/non_rg.py -eoc len -season $ses
    python3 utils/non_rg.py -eoc concepts -season $ses
    python3 utils/non_rg.py -eoc entities -season $ses
    python3 utils/prep_eval_res.py -season $ses

    echo "${ses} season done!!!"
    echo "=========================================================="
    echo " "
done
