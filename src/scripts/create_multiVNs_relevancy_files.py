import os
import sys
import pickle
import pandas as pd
import numpy as np
#import word2vec as wv
from tqdm import tqdm

from scripts.create_multiVNs_feature_files import create_sentence_df
from scripts.create_multiVNs_relational_files import save_output_dict as save_output_arr


def get_new_dataframe_names(df_path):
    file_name = df_path.rsplit('/')[-1]
    return '{}_relevancy_' + file_name


def create_verb_relevancy(df, sentence_df):
    verb_union_arr = np.zeros([len(df), len(sentence_df)])
    verb_intersection_arr = np.zeros([len(df), len(sentence_df)])

    video_verb_list = list(df.parsed_verb_classes.apply(lambda x: frozenset(x)))
    sentence_verb_list = list(sentence_df.verb_class.apply(lambda x: frozenset(x)))

    for i in tqdm(range(len(df))):
        for j in range(len(sentence_df)):
            int_arr = len(video_verb_list[i].intersection(sentence_verb_list[j]))
            uni_arr = len(video_verb_list[i].union(sentence_verb_list[j]))

            verb_intersection_arr[i][j] = int_arr
            verb_union_arr[i][j] = uni_arr
    verb_iou = verb_intersection_arr / verb_union_arr
    verb_iou[np.isinf(verb_iou)] = 0
    verb_iou[np.isnan(verb_iou)] = 0
    return verb_iou


def create_noun_relevancy(df, sentence_df):
    video_unique_noun_sets = df.parsed_noun_classes.apply(lambda x: frozenset(x)).unique()  # REM
    sentence_unique_noun_sets = sentence_df.noun_class.apply(lambda x: frozenset(x)).unique()  #REM
    noun_intersection_arr = np.zeros([len(video_unique_noun_sets), len(sentence_unique_noun_sets)])
    noun_union_arr = np.zeros([len(video_unique_noun_sets), len(sentence_unique_noun_sets)])
    print(len(video_unique_noun_sets), len(sentence_unique_noun_sets))
    for i in tqdm(range(len(video_unique_noun_sets))):
        for j in range(len(sentence_unique_noun_sets)):
            int_arr = len(video_unique_noun_sets[i].intersection(sentence_unique_noun_sets[j]))
            uni_arr = len(video_unique_noun_sets[i].union(sentence_unique_noun_sets[j]))

            noun_intersection_arr[i][j] = int_arr
            noun_union_arr[i][j] = uni_arr

    video_noun_set_to_idx_dict = {video_unique_noun_sets[i]: i for i in range(len(video_unique_noun_sets))}
    sentence_noun_set_to_idx_dict = {sentence_unique_noun_sets[i]: i for i in range(len(sentence_unique_noun_sets))}
    df['noun_set_idx'] = df.parsed_noun_classes.apply(lambda x: video_noun_set_to_idx_dict[frozenset(x)])
    sentence_df['noun_set_idx'] = sentence_df.noun_class.apply(lambda x: sentence_noun_set_to_idx_dict[frozenset(x)])

    expanded_noun_intersection_arr = np.zeros([len(df), len(sentence_df)])
    expanded_noun_union_arr = np.zeros([len(df), len(sentence_df)])
    video_noun_set_idxs = {i: df.iloc[i]['noun_set_idx'] for i in range(len(df))}
    sentence_noun_set_idxs = {i: sentence_df.iloc[i]['noun_set_idx'] for i in range(len(sentence_df))}

    for i in tqdm(range(len(df))):
        i_idx = video_noun_set_idxs[i]
        for j in range(len(sentence_df)):
            j_idx = sentence_noun_set_idxs[j]
     
            expanded_noun_intersection_arr[i][j] = noun_intersection_arr[i_idx][j_idx]
            expanded_noun_union_arr[i][j] = noun_union_arr[i_idx][j_idx]
    noun_iou = expanded_noun_intersection_arr / expanded_noun_union_arr
    noun_iou[np.isinf(noun_iou)] = 0
    noun_iou[np.isnan(noun_iou)] = 0
    return noun_iou


def main(args):
    df = pd.read_pickle(args.dataframe)
    dataframe_name = get_new_dataframe_names(args.dataframe)
    sentence_df = create_sentence_df(df)
    #sentence_df.to_pickle("train_df.pkl")
    print(df)
    print(sentence_df)

    if args.caption or args.noun:
        noun_iou = create_noun_relevancy(df, sentence_df)
        if args.noun:
            save_output_arr(noun_iou, args.out_dir, dataframe_name, 'noun_prova')
    if args.caption or args.verb:
        verb_iou = create_verb_relevancy(df, sentence_df)
        if args.verb:
            save_output_arr(verb_iou, args.out_dir, dataframe_name, 'verb_prova')
    if args.caption:
        average_caption_iou = (verb_iou + noun_iou) / 2
        save_output_arr(average_caption_iou, args.out_dir, dataframe_name, 'caption_prova')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser('Creates relevancy feature files for MMEN/JPoSE')
    parser.add_argument('dataframe', type=str, help='Annotations dataframe to create feature files from')
    parser.add_argument('--out-dir', type=str, help='Output directory to save files to. [./]')
    parser.add_argument('--caption', action='store_true', help='Create caption relational file. [True]')
    parser.add_argument('--no-caption', action='store_false', dest='caption', help='Do not create caption relevancy file.')
    parser.add_argument('--verb', action='store_true', help='Create verb relevancy file. [True]')
    parser.add_argument('--no-verb', action='store_false', dest='verb', help='Do not create verb relevancy file.')
    parser.add_argument('--noun', action='store_true', help='Create noun relevancy file. [True]')
    parser.add_argument('--no-noun', action='store_false', dest='noun', help='Do not noun caption relevancy file.')

    parser.set_defaults(
            out_dir='./',
            caption=True,
            verb=True,
            noun=True
    )
    main(parser.parse_args())
