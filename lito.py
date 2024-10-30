# LANGUAGE-INTERFACED TABULAR OVERSAMPLING VIA PROGRESSIVE IMPUTATION AND SELF AUTHENTICATION
# paper experiment reproduction
# Soowon.oh (osw5144@gmail.com)

from be_great import GReaT
import pandas
import math
import os
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import umap

CODE_TEST = False

# finetune TLM conducted on colab
def finetune_TLM(dataset: str):
    data = load_dataset_as_df(dataset)
    model = GReaT(llm='pretrained', batch_size=32, epochs=300, save_steps=400000, logging_steps=100)
    model.fit(data)
    os.makedirs("finetuned_model", exist_ok=True)
    model.save(f'finetuned_model/{dataset}')

def load_dataset_as_df(dataset: str) -> pandas.DataFrame:
    data = pandas.read_csv(f'dataset/{dataset}.csv')
    return data

def load_TLM(dataset: str):
    model = GReaT.load_from_dir(f'finetuned_model/{dataset}')
    model.model.config.output_attentions = True
    return model

def make_sample_imbalanced(df: pandas.DataFrame, alpha: float) -> list:
    # assume that the last column name is for classification
    class_col_name = df.columns[-1]
    df = df.sort_values(by=class_col_name)
    
    classes = df[class_col_name].unique()
    class_partition_dict = {}
    class_cnt = [] # used for sorting
    for class_val in classes:
        df_par = df[df[class_col_name] == class_val]
        class_partition_dict[class_val] = df_par
        class_cnt.append((len(df_par),class_val)) # (class count, class value)
    class_cnt = sorted(class_cnt, reverse=True) # sorted by class counts in descending order
    
    ret = [class_partition_dict[class_cnt[0][1]]] # pre-filled majority class df
    # calculate at most size for each class
    gamma = 1/alpha # alpha is imbalance ratio between majority sample and most minority sample
    major_count = class_cnt[0][0]
    for i in range(2,len(class_cnt)+1): # 1-index, and no need to process for major class(index=1)
        cur_alpha = gamma**(-(i-1)/(len(class_cnt)-1))
        atmost_count = math.ceil(major_count/cur_alpha)
        
        count,name = class_cnt[i-1]
        # NOTE that atmost_count still guarantees the sample majority order,
        # because the lower bound of each sample count is itself, and already ordered by itself.
        class_df = class_partition_dict[name]
        if count > atmost_count:
            # drop samples
            class_df = class_df.iloc[:atmost_count]
        else:
            # already sample counts is lower than atmost_count, keep it same
            pass
        ret.append(class_df)
    return ret

from be_great.great_utils import _convert_tokens_to_text, _convert_text_to_tabular_data
import torch
import torch.nn.functional as F
# for attention weight visualization
import matplotlib.pyplot as plt
import seaborn as sns

MAXIMUM_GEN_TRIAL = 1000000 # at most major sample 10K. give chances 100 times per samples

def batch_self_authentication(df_gen, minor_class:str, model, device='cuda') -> list:
    # len(df_gen) is batch size
    columns = df_gen.columns
    batch_texts = []
    for batch_idx in range(len(df_gen)):
        text = ", ".join([f"{col} is {df_gen.iloc[batch_idx][col]}" for col in columns[:-1]])
        batch_texts.append(text)
    input = model.tokenizer(batch_texts, padding=True, padding_side='left', return_tensors='pt')
    model.model.to(device)
    output = model.model.generate(input_ids=torch.tensor(input.input_ids).to(device),
                                  max_length=1000,
                                  do_sample=True,
                                  temperature=0.7,
                                  pad_token_id=50256,
                                  output_scores=True,
                                  return_dict_in_generate=True)
    
    ret_list = []
    # to calculate confidence for target label (minor class)
    # use heuristic. next of "is" token is classification
    # put space front as same condition following "is"
    minor_class_tokens = model.tokenizer(f" {minor_class}").input_ids
    for batch_idx in range(len(df_gen)):
        tok_id = 0
        confidence = 1.0 # init prob
        calc_start = False
        for step, logits in enumerate(output.scores): # during generation steps, calculating confidence
            # Convert logits to probabilities using softmax
            probs = F.softmax(logits[batch_idx], dim=-1)
            
            # Get the most probable token at each step
            most_probable_token_id = torch.argmax(logits[batch_idx], dim=-1).item()
            most_probable_token = model.tokenizer.decode([most_probable_token_id])
            
            if 'endoftext' in most_probable_token:
                break # don't count endoftext token prob
            
            if calc_start:
                if tok_id < len(minor_class_tokens):
                    confidence *= probs[minor_class_tokens[tok_id]].item() # partial minor tok prob
                    tok_id += 1
                else:
                    # minor class token is shorter than predicted class token.
                    # just keep confidence
                    pass
            if 'is' in most_probable_token:
                calc_start = True # from next step, calc prob
        if not calc_start:
            confidence = 0.0
        
        text_data = _convert_tokens_to_text(output.sequences[batch_idx:batch_idx+1],model.tokenizer)
        authen_result_df = _convert_text_to_tabular_data(text_data,columns)
        good = authen_result_df.iloc[0,-1] == minor_class
        ret_list.append([good,confidence])
    return ret_list

def oversample_class_cond(model, class_val, columns, n_samples: int, device='cuda', batch_size: int = 64) -> pandas.DataFrame:
    # LITO-C: full class-conditioned
    class_col_name = columns[-1]
    
    gen_count = 0
    trial_count = 0 # to avoid infinite-loop
    pbar = tqdm(total=n_samples)
    
    ret_df = pandas.DataFrame()
    while gen_count < n_samples and trial_count < MAXIMUM_GEN_TRIAL:
        current_batch_size = min(n_samples - gen_count, batch_size) # this will be active at the last batch
        trial_count += current_batch_size
        
        batch_sample_texts = []
        for _ in range(current_batch_size):
            text = f"{class_col_name} is {class_val}" # minor class
            batch_sample_texts.append(text)
        assert len(batch_sample_texts) == current_batch_size
    
        input = model.tokenizer(batch_sample_texts, return_tensors='pt')
        model.model.to(device)
        sampled_output = model.model.generate(input_ids=torch.tensor(input.input_ids).to(device),
                                              max_length=1000,
                                              do_sample=True,
                                              temperature=0.7,
                                              pad_token_id=50256)
        text_data = _convert_tokens_to_text(sampled_output,model.tokenizer)
        df_gen = _convert_text_to_tabular_data(text_data,columns)
        
        placeholder_index = df_gen[df_gen.isin(['placeholder'])].dropna(how='all').index.tolist()
        nan_index = df_gen[df_gen.isna().any(axis=1)].index.tolist()
        drop_batch_idx = set(placeholder_index + nan_index)
        df_gen = df_gen.drop(drop_batch_idx)
        assert len(df_gen) == current_batch_size - len(drop_batch_idx), f"{len(df_gen)}, {current_batch_size - len(drop_batch_idx)}"

        if len(df_gen) > 0:
            authen_ret = batch_self_authentication(df_gen, class_val, model)
            authen_idx = 0 # from top
            for batch_idx in range(current_batch_size):
                if batch_idx in drop_batch_idx:
                    continue
                good,_ = authen_ret[authen_idx]

                if good:
                    ret_df = pandas.concat([ret_df,df_gen.iloc[authen_idx:authen_idx+1]],ignore_index=True)
                    gen_count += 1
                    pbar.update(1)
                else: # not good
                    pass
                authen_idx += 1
        
        # if reach maximum gen trial, just takes generated and go next
        if trial_count >= MAXIMUM_GEN_TRIAL:
            # assert False, "reach to maximum gen trial before generation complete"
            print("@@@ trial_count reaches. not address imbalanced data")
    return ret_df

# model run in batch
def get_sorted_imputable_feature_list_from_sample(model, major_sample: pandas.DataFrame, n_samples: int, device='cuda') -> list:
    columns = major_sample.columns
    class_col_name = columns[-1]
    
    # get majority sample for borderline sampling
    text_list = []
    for _ in range(n_samples):
        sample_seed = major_sample.sample(n=1) # get random sample
        class_text = f"{class_col_name} is {sample_seed[class_col_name].item()}"
        feature_text = ", ".join([f"{col} is {sample_seed[col].item()}" for i,col in enumerate(columns[:-1])])
        text = class_text + ", " + feature_text
        text_list.append(text)
    
    input = model.tokenizer(text_list, padding=True, return_tensors='pt')
    model.model.to(device) # move model to device
    input = {key: value.to(device) for key, value in input.items()}
    model.model.eval() # evaluation mode
    outputs = model.model(**input)
    
    ret_list = []
    for batch_idx in range(n_samples):
        # get the last layer's attention weight
        last_attention_weight = outputs.attentions[-1][batch_idx] # shape = (num_heads,seq_len,seq_len)
        ### VISUALIZATION CODE ###
        # tokens = model.tokenizer.convert_ids_to_tokens(input.input_ids[0])
        # fig,axes = plt.subplots(3,4, figsize=(20,15))
        # fig.suptitle(f"Attention Heads for [{text}]", fontsize=16)
        # for i in range(12):  # We have 12 attention heads
        #     head_attention = last_attention_weight[i].detach().numpy()  # Get the ith head's attention weights
        #     ax = axes[i // 4, i % 4]  # Determine subplot position
        #     sns.heatmap(head_attention, xticklabels=tokens, yticklabels=tokens, cmap="viridis", ax=ax)
        #     # Set title and labels
        #     ax.set_title(f"Head {i+1}")
        #     ax.set_xlabel("Query Tokens")
        #     ax.set_ylabel("Key Tokens")
        #     ax.tick_params(axis='x', rotation=90)
        #     ax.tick_params(axis='y', rotation=0)
        # plt.tight_layout(rect=[0, 0, 1, 0.96])  # Make space for the suptitle
        # plt.show(
        # calculate importance score for features(columns)
        column_seq_ranges = list()
        token_id_seq = input['input_ids'][0].tolist()
        # NOTE: to partition with column tokens, use tokenizer heuristic.
        # token(11) is ', '. so I used this token as deliminator.
        left_window = 0
        for t in range(len(token_id_seq)):
            if token_id_seq[t] == 11: # deliminator(', ')
                column_seq_ranges.append((left_window,t-1))
                left_window = t+1
        column_seq_ranges.append((left_window,len(token_id_seq)-1)) # last column consumption
        assert len(column_seq_ranges) == len(columns)
        importance_scores = list()
        for col_idx in range(1,len(columns)): # we don't handle class colmun
            # score = 0
            # for h in range(last_attention_weight.shape[0]): # num heads
            #     for c in range(column_seq_ranges[col_idx][0],column_seq_ranges[col_idx][1]+1): # feature tok range
            #         for b in range(column_seq_ranges[0][0],column_seq_ranges[0][1]+1): # class tok range
            #             score += last_attention_weight[h][c][b].item()
            # for speedup
            score = last_attention_weight[
                :,
                column_seq_ranges[col_idx][0]:column_seq_ranges[col_idx][1]+1,
                column_seq_ranges[0][0]:column_seq_ranges[0][1]+1].sum()
            importance_scores.append((score,col_idx-1)) # score , data_column_idx

        ### VISUALIZATION CODE ###
        # only_imp_score = [sc[0] for sc in importance_scores]
        # bars = plt.bar(data_columns[:-1], only_imp_score)
        # plt.xlabel('Features')
        # plt.ylabel('Importance Score')
        # for bar in bars:
        #     yval = bar.get_height()
        #     plt.text(bar.get_x() + bar.get_width()/2, yval, round(yval,2), ha='center', va='bottom')
        # plt.show()

        importance_scores = sorted(importance_scores) # sort by score in ascending order
        feature_list = [[sample_seed[columns[entry[1]]].item(),entry[1]] for entry in importance_scores]
        ret_list.append([feature_list,0.0]) # [feature_list, init_prob]
    return ret_list

# to see the pregress, applied tqdm
def oversample_borderline(model, class_val, major_sample, n_samples: int, device='cuda', topk: int = 4, batch_size: int = 128) -> pandas.DataFrame:
    # LITO-B: feature condition + class condition (progressive imputation by default)
    columns = major_sample.columns
    class_col_name = columns[-1]
    
    gen_count = 0
    trial_count = 0 # to avoid infinite-loop
    pbar = tqdm(total=n_samples)
    
    ret_df = pandas.DataFrame()

    batch_sample_feature_imputable = []
    while gen_count < n_samples and trial_count < MAXIMUM_GEN_TRIAL:
        current_batch_size = min(n_samples - gen_count, batch_size) # this will be active at the last batch
        trial_count += current_batch_size
        
        if len(batch_sample_feature_imputable) < current_batch_size:
            add_feature_list = get_sorted_imputable_feature_list_from_sample(model, major_sample,\
                current_batch_size - len(batch_sample_feature_imputable))
            batch_sample_feature_imputable += add_feature_list
        assert len(batch_sample_feature_imputable) == current_batch_size, f"{len(batch_sample_feature_imputable)}, {current_batch_size}"

        batch_sample_texts = []
        for batch_idx in range(current_batch_size):
            feature_list,_ = batch_sample_feature_imputable[batch_idx]
            assert len(feature_list) > 0
            
            # impute top-k features
            impute_count = 0
            while len(feature_list) > 0 and impute_count < topk:
                feature_list.pop()
                impute_count += 1
            
            text = f"{class_col_name} is {class_val}" # minor class
            if len(feature_list) > 0:
                feature_text = ", ".join([f"{columns[entry[1]]} is {entry[0]}" for entry in feature_list])
                text = text + ", " + feature_text
            batch_sample_texts.append(text)
        assert len(batch_sample_texts) == current_batch_size
        
        input = model.tokenizer(batch_sample_texts, padding=True, padding_side='left', return_tensors='pt')
        model.model.to(device)
        sampled_output = model.model.generate(input_ids=torch.tensor(input.input_ids).to(device),
                                              max_length=1000,
                                              do_sample=True,
                                              temperature=0.7,
                                              pad_token_id=50256)
        text_data = _convert_tokens_to_text(sampled_output,model.tokenizer)
        df_gen = _convert_text_to_tabular_data(text_data,columns)

        placeholder_index = df_gen[df_gen.isin(['placeholder'])].dropna(how='all').index.tolist()
        nan_index = df_gen[df_gen.isna().any(axis=1)].index.tolist()
        drop_batch_idx = set(placeholder_index + nan_index) # this will be reused at parsing authen_ret
        df_gen = df_gen.drop(drop_batch_idx)
        assert len(df_gen) == current_batch_size - len(drop_batch_idx), f"{len(df_gen)}, {current_batch_size - len(drop_batch_idx)}"
        
        if len(df_gen) > 0:
            authen_ret = batch_self_authentication(df_gen, class_val, model)
            authen_idx = 0 # from top
            for batch_idx in range(current_batch_size):
                if batch_idx in drop_batch_idx:
                    continue
                good,cur_prob = authen_ret[authen_idx]
                feature_list,prev_prob = batch_sample_feature_imputable[batch_idx]

                if good:
                    ret_df = pandas.concat([ret_df,df_gen.iloc[authen_idx:authen_idx+1]],ignore_index=True)
                    gen_count += 1
                    pbar.update(1)
                    drop_batch_idx.add(batch_idx) # drop
                else: # not good
                    if cur_prob <= prev_prob:
                        drop_batch_idx.add(batch_idx) # drop
                    else: # still have chance to get a candidate
                        if len(feature_list) == 0: # no feature any more
                            drop_batch_idx.add(batch_idx) # drop
                        else:
                            # do next iter
                            batch_sample_feature_imputable[batch_idx][1] = cur_prob # current prob update
                authen_idx += 1
        
        # make sequential batch_sample_feature_imputable without intermediate blanks
        new_batch_sample_feature_imputable = []
        for batch_idx in range(current_batch_size):
            if batch_idx not in drop_batch_idx:
                new_batch_sample_feature_imputable.append(batch_sample_feature_imputable[batch_idx])
        # pack the rest as None
        batch_sample_feature_imputable = new_batch_sample_feature_imputable
        
        # if reach maximum gen trial, just takes generated and go next
        if trial_count >= MAXIMUM_GEN_TRIAL:
            # assert False, "reach to maximum gen trial before generation complete"
            print("@@@ trial_count reaches. not address imbalanced data")
    return ret_df

def oversample(model, samples, sample_algo: str, topk: int, device='cuda', batch_size: int = 32) -> pandas.DataFrame: # return oversampled samples
    # assume that samples is sorted in class majority
    data_columns = samples[0].columns
    major_sample = samples[0]
    major_count = len(samples[0])
    
    ret_gen = pandas.DataFrame()
    if sample_algo == 'lito-c':
        # LITO-C: full class-conditioned
        for i in range(1,len(samples)): # skip majority class(idx=0)
            class_val = samples[i].iloc[0,-1]
            req_count = 1 if CODE_TEST else major_count - len(samples[i]) # minority sample count offset
            df_gen = oversample_class_cond(model, class_val, data_columns, req_count)
            # Remove rows with flawed numerical values but keep NaNs
            for i_num_cols in model.num_cols:
                coerced_series = pandas.to_numeric(
                    df_gen[i_num_cols], errors="coerce"
                )
                df_gen = df_gen[
                    coerced_series.notnull() | df_gen[i_num_cols].isna()
                ]
            # Convert numerical columns to float
            df_gen[model.num_cols] = df_gen[model.num_cols].astype(float)
            
            ret_gen = pandas.concat([ret_gen,df_gen],ignore_index=True)
    elif sample_algo == 'lito-b':
        for i in range(1,len(samples)): # skip majority class(idx=0)
            class_val = samples[i].iloc[0,-1]
            req_count = 1 if CODE_TEST else major_count - len(samples[i]) # minority sample count offset
            df_gen = oversample_borderline(model, class_val, major_sample, req_count)
            # Remove rows with flawed numerical values but keep NaNs
            for i_num_cols in model.num_cols:
                coerced_series = pandas.to_numeric(
                    df_gen[i_num_cols], errors="coerce"
                )
                df_gen = df_gen[
                    coerced_series.notnull() | df_gen[i_num_cols].isna()
                ]
            # Convert numerical columns to float
            df_gen[model.num_cols] = df_gen[model.num_cols].astype(float)
            
            ret_gen = pandas.concat([ret_gen,df_gen],ignore_index=True)
    else:
        assert sample_algo == 'vanilla'
        # do nothing

    # after oversampling, aggregate all samples
    ret = pandas.DataFrame()
    for sample in samples:
        ret = pandas.concat([ret,sample], ignore_index=True)
    if len(ret_gen) > 0:
        ret = pandas.concat([ret,ret_gen], ignore_index=True)
    return ret

from sklearn import tree
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import f1_score, balanced_accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

def eval_ML_efficiency(clf, train_df, test_df, binary: bool, pos_label: str) -> tuple: # return (f1,bacc)
    num_classes = len(train_df[train_df.columns[-1]].unique())

    # NOTE: if feature has descrete format except for class,
    #   it should be transformed to numeric format for ML model.
    # NOTE: for XGBoost, class also should be transformed with label encoding
    chunk_df = pandas.concat([train_df, test_df], ignore_index=True)
    for col in train_df.columns:
        encode_require = False
        for uniq_val in chunk_df[col].unique():
            if isinstance(uniq_val, str):
                encode_require = True
        if encode_require:
            le = LabelEncoder()
            # all unique should have string, but sometimes synthetic outlier happens
            # make all unique values to string forcely
            chunk_df[col] = chunk_df[col].astype(str)
            chunk_df[col] = le.fit_transform(chunk_df[col])
            if col == train_df.columns[-1]: # class column
                # need to encode pos_label
                for cname,enc_id in zip(le.classes_, range(len(le.classes_))):
                    if cname == pos_label:
                        pos_label = enc_id

    train_df = chunk_df.iloc[:len(train_df)]
    test_df = chunk_df.iloc[len(train_df):]

    # assume that the last column is class column
    train_X = train_df.iloc[:,:-1]
    train_Y = train_df.iloc[:,-1:]
    test_X = test_df.iloc[:,:-1]
    test_Y = test_df.iloc[:,-1:]

    # NOTE: if feature has descrete format except for class,
    #   it should be transformed to numeric format for ML model.
    # to have same encoding, aggregate train_X and test_X
    chunk_X = pandas.concat([train_X, test_X], ignore_index=True)
    for col in train_X.columns:
        encode_require = False
        for uniq_val in chunk_X[col].unique():
            if isinstance(uniq_val, str):
                encode_require = True
        if encode_require:
            le = LabelEncoder()
            chunk_X[col] = le.fit_transform(chunk_X[col])

    train_X = chunk_X.iloc[:len(train_X)]
    test_X = chunk_X.iloc[len(train_X):]

    # train ML model
    clf = clf.fit(train_X, train_Y)
    pred_Y = clf.predict(test_X)
    
    if binary: # binary classification
        # print(classification_report(test_Y,pred_Y))
        f1 = f1_score(test_Y, pred_Y, pos_label=pos_label)
        bacc = balanced_accuracy_score(test_Y, pred_Y)
    else: # multi-class classification
        assert num_classes > 2
        f1 = f1_score(test_Y, pred_Y, average='weighted')
        bacc = balanced_accuracy_score(test_Y, pred_Y)
    return f1,bacc

def evaluate(train_df, test_df, f1_list, bacc_list, binary: bool, pos_label: str):
    clfs = [
        tree.DecisionTreeClassifier(max_depth=32, criterion='gini'),
        LogisticRegression(solver='lbfgs', max_iter=1000, penalty='l2')
    ]
    if binary:
        clfs += [
            AdaBoostClassifier(n_estimators=100, learning_rate=1.0),
            MLPClassifier(hidden_layer_sizes=(100,100), max_iter=200, alpha=0.0001)
        ]
    else:
        clfs += [
            RandomForestClassifier(n_estimators=100, max_depth=8),
            XGBClassifier(objective='multi:softmax', max_depth=5, learning_rate=1.0, n_estimators=100)
        ]
    for clf in clfs:
        f1,bacc = eval_ML_efficiency(clf,train_df,test_df,binary,pos_label)
        f1_list.append(f1)
        bacc_list.append(bacc)

DATASET = ['diabetes','obesity']
OVERSAMPLING_COUNT = {
    'diabetes':4,
    'obesity':2
}
ML_EVAL_COUNT = 5
TEST_RATIO = {
    'diabetes':0.3,
    'obesity':0.2
}
MULTI_CLASS_DATASET = {'obesity','satimage'}
SAMPLE_ALGOS = ['vanilla', 'lito-b', 'lito-c', 'great']
DATASET_POS_LABEL = { # only for binary classification
    'diabetes':'tested_positive',
    'sick':'sick'
}
DATASET_ALPHA = {
    'diabetes':{
        'mild':10,
        'extreme':20
    },
    'obesity':{
        'mild':10,
        'extreme':100
    },
}
DATASET_TOPK = {
    'diabetes':1,
    'obesity':4
}

static_reducer = None

import numpy as np
def run(train_samples: list, test_df, dataset: str, sample_algo: str, alpha: float, file, reducer = None, le_list = None):
    print(f"run with {dataset},{sample_algo},alpha={alpha}")
    file.write(f"run with {dataset},{sample_algo},alpha={alpha}\n")

    binary = False if dataset in MULTI_CLASS_DATASET else True
    model = load_TLM(dataset)
    f1_list,bacc_list = list(),list()
    for oversample_idx in range(OVERSAMPLING_COUNT[dataset]):
        if sample_algo != 'great':
            train_df = oversample(model, train_samples, sample_algo, topk=DATASET_TOPK[dataset])
        else:
            assert sample_algo == 'great', f"invalid sample algorithm: {sample_algo}"
            major_count = len(train_samples[0])
            gen_count = 0
            for i in range(1,len(train_samples)):
                gen_count += (major_count - len(train_samples[i]))
            gen_df = model.sample(n_samples=gen_count, max_length=1000) # no start col => class col and dist
            train_df = pandas.DataFrame()
            for sample in train_samples:
                train_df = pandas.concat([train_df,sample], ignore_index=True)
            train_df = pandas.concat([train_df,gen_df], ignore_index=True)

        num_classes = len(train_df[train_df.columns[-1]].unique())
        if num_classes < 2:
            print('sampling miss, only one class sampled. drop sample')
            continue
        # train_df.to_csv(f'train_df_{alpha}_{sample_algo}.csv', index=False)
        
        ### VISUALIZATION CODE ###
        # vis_train_df = train_df.copy()
        # print(len(vis_train_df))
        # for col in le_list:
        #     vis_train_df[col] = vis_train_df[col].astype(str)
        #     vis_train_df[col] = le_list[col].transform(vis_train_df[col])
        # # drop samples that should not be string
        # for col in vis_train_df.columns:
        #     if col not in le_list:
        #         mask = vis_train_df[col].apply(lambda x: isinstance(x,str))
        #         vis_train_df = vis_train_df[~mask]
        
        # train_X = vis_train_df.iloc[:,:-1]
        # train_Y = vis_train_df.iloc[:,-1:]
        # if sample_algo == 'vanilla':
        #     reducer.fit(train_X)
        # embedding = reducer.transform(train_X)
        
        # color = train_Y.iloc[:,0].tolist()
        # plt.scatter(embedding[:,0],embedding[:,1], c=color, s=1, cmap='viridis')
        # plt.title('UMAP')
        # plt.savefig(f'umap_{dataset}_{sample_algo}_{alpha}_{oversample_idx}.png')
        # plt.clf()

        # evaluation with ML models
        for _ in range(ML_EVAL_COUNT):
            pos_label = DATASET_POS_LABEL[dataset] if binary else ""
            evaluate(train_df, test_df, f1_list, bacc_list, binary, pos_label)
    # afterall, mean and std dev for f1,bacc
    f1_mean,f1_stddev = np.mean(f1_list),np.std(f1_list)
    f1_stderr = f1_stddev / math.sqrt(len(f1_list))
    bacc_mean,bacc_stddev = np.mean(bacc_list),np.std(bacc_list)
    bacc_stderr = bacc_stddev / math.sqrt(len(bacc_list))
    print(f"f1 score: {f1_mean},{f1_stderr}")
    file.write(f"f1 score: {f1_mean},{f1_stderr}\n")
    print(f"bacc: {bacc_mean},{bacc_stderr}")
    file.write(f"bacc: {bacc_mean},{bacc_stderr}\n\n")
    print(f1_list)
    print(bacc_list)
    file.flush()

import warnings
if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    result_file = 'result.txt'
    file = open(result_file, mode='w')

    for dataset in DATASET:
        df = load_dataset_as_df(dataset)
        for mildness in DATASET_ALPHA[dataset]:
            alpha = DATASET_ALPHA[dataset][mildness]
            # make base dataframe for train, test
            train_df, test_df = train_test_split(df, test_size=TEST_RATIO[dataset])
            imbalanced_train_samples = make_sample_imbalanced(train_df, alpha)
            # for umap visualize
            reducer = umap.UMAP(random_state=1, n_neighbors=15, min_dist=0.25)
            le = LabelEncoder()
            le_list = {}
            for col in train_df.columns:
                encode_require = False
                for uniq_val in train_df[col].unique():
                    if isinstance(uniq_val, str):
                        encode_require = True
                if encode_require:
                    train_df[col] = train_df[col].astype(str)
                    le_list[col] = LabelEncoder()
                    le_list[col].fit(train_df[col])
                    # all unique should have string, but sometimes synthetic outlier happens
                    # make all unique values to string forcely
                
            for sample_algo in SAMPLE_ALGOS:
                run(imbalanced_train_samples,test_df,dataset,sample_algo,alpha,file,reducer,le_list)
    
    file.close()