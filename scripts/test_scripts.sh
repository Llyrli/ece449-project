cd ..

# raw_data_path: the path of raw data
# processed data path: where you put the processed data (including the tokenized data and text data after train-test-split)
# model_list: put the model you want to train/eval with ','
# predict_bool: if you want to predict and test the model result on specific data

python main.py \
        --raw_data_path ./data \
        --process_data_path ./processed_data \
        --model_list bayes,random_forest,pytorch_mlp,bert \
        --predict_bool \
        --test_dataset \
        --test_dataset_path './ECE449_dataset'