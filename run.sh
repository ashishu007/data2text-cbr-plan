# change the season inside the file

ses=$1
echo "Season: $ses"

echo "Creating Imp Player Classifier Data..."
python3 imp_players_clf_data.py -season $ses

echo "Creating Imp Player Classifier..."
python3 player_clf/main.py -season $ses

echo "Creating Player Popularity Data..."
python3 calc_popularity.py -season $ses

echo "Extracting Concepts from Gold Summaries..."
python3 extract_from_gold.py -season $ses

echo "Creating Case-Base for ${ses} season..."
python3 create_cb.py -side both -season $ses -pop

echo "Generating Concepts on Test Set..."
python3 gen_concepts.py -pop -season $ses

echo "Evaluating..."
python3 non_rg.py len $ses
python3 non_rg.py concepts $ses
python3 non_rg.py entities $ses
