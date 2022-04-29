import os
import sys
import pickle
import pandas as pd
import numpy as np
#import word2vec as wv
from tqdm import tqdm

from scripts.create_feature_files import create_sentence_df, pre_process_word_idx, fix_sentence_df


def create_no_class_sentence_df(df):
    df['action_class'] = df.apply(lambda x: '{}_{}'.format(x.verb_class, x.noun_class), axis=1)
    sentences = {}
    indices = {}
    classes = {}
    verb_class = {}
    noun_class = {}
    verbs = {}
    nouns = {}
    unique_classes = df.action_uid.unique()
    for i, class_ in tqdm(enumerate(unique_classes), total=len(unique_classes)):
        classes[i] = class_
        subset_df = df[df.action_uid.apply(lambda x: x == class_)]
        indices[i] = subset_df.index[0]
        sentences[i] = subset_df.iloc[0].narration
        verb_class[i] = subset_df.iloc[0].verb_class
        noun_class[i] = subset_df.iloc[0].all_noun_classes
        verbs[i] = subset_df.iloc[0].verb
        nouns[i] = subset_df.iloc[0].all_nouns
    sentence_df = pd.DataFrame([sentences]).T
    sentence_df.columns = ['sentence']
    sentence_df['action_uid'] = pd.Series(classes)
    sentence_df['index'] = pd.Series(indices)
    sentence_df['verb_class'] = pd.Series(verb_class)
    sentence_df['noun_class'] = pd.Series(noun_class)
    sentence_df['verb'] = pd.Series(verbs)
    sentence_df['nouns'] = pd.Series(nouns)
    return sentence_df


def get_new_dataframe_names(df_path, class_):
    file_name = df_path.rsplit('/')[-1]
    class_name = '' if class_ else 'no-class_'
    return class_name + '{}_relational_' + file_name

def save_output_dict(out_dict, out_dir, df_name, type_):
    out_path = '/'.join([out_dir, df_name.format(type_)])
    with open(out_path, 'wb') as out_f:
        pickle.dump(out_dict, out_f)


def create_PoS_rel_dict(df, sentence_df, type_='action_class', all_nouns=False):
    rel_dict = {}
    if all_nouns and type_.startswith('action'):
        vid_to_class_dict, class_to_vid_dict, vid_to_classes_dict = create_rel_dict(df, type_, all_nouns)
        sent_to_class_dict, class_to_sent_dict, sent_to_classes_dict = create_rel_dict(sentence_df, type_, all_nouns)
    else:
        vid_to_class_dict, class_to_vid_dict = create_rel_dict(df, type_, all_nouns)
        sent_to_class_dict, class_to_sent_dict = create_rel_dict(sentence_df, type_, all_nouns)
    rel_dict['vid2class'] = vid_to_class_dict
    rel_dict['class2vid'] = class_to_vid_dict
    rel_dict['sent2class'] = sent_to_class_dict
    rel_dict['class2sent'] = class_to_sent_dict
    if all_nouns and type_.startswith('action'):
        rel_dict["vid2classes"] = vid_to_classes_dict
        rel_dict["sent2classes"] = sent_to_classes_dict
    return rel_dict


def create_rel_dict(df, column, all_nouns=False):
    mod_to_class_dict = {}
    class_to_mod_dict = {}
    if all_nouns:
        mod_to_classes_dict = {}

    for i, row in tqdm(df.iterrows(), total=len(df)):
        if not column.startswith('noun') or not all_nouns:
            mod_to_class_dict[i] = row[column]
            if row[column] not in class_to_mod_dict:
                try:
                    class_to_mod_dict[row[column]] = [i]
                except:
                    import bpdb; bpdb.set_trace()
            else:
                class_to_mod_dict[row[column]].append(i)

            if all_nouns:
                try:
                    mod_to_classes_dict[i] = row["all_noun_classes"]
                except:
                    mod_to_classes_dict[i] = row["noun_class"]
        else:
            # here we want to enter when column='noun_*' (-> computing the nouns pkl) and all_nouns;
            # want to use *all* the noun classes,
            try:
                all_cls = row["all_noun_classes"]
            except:
                all_cls = row["noun_class"]

            mod_to_class_dict[i] = all_cls
            for c in all_cls:
                if c not in class_to_mod_dict:
                    try:
                        class_to_mod_dict[c] = [i]
                    except:
                        import bpdb; bpdb.set_trace()
                else:
                    class_to_mod_dict[c].append(i)

    if all_nouns and column.startswith('action'):
        return mod_to_class_dict, class_to_mod_dict, mod_to_classes_dict
    else:
        return mod_to_class_dict, class_to_mod_dict


def main(args):
    df = pd.read_pickle(args.dataframe)
    dataframe_name = get_new_dataframe_names(args.dataframe, args.class_)
    if not args.class_:
        df, v2idx, n2idx = pre_process_word_idx(df)
        sentence_df = create_sentence_df(df, args.all_noun_classes)
        #if not args.all_noun_classes:
        sentence_df['noun'] = sentence_df.nouns.apply(lambda x: x[0])
        #else:
        #    sentence_df['noun'] = sentence_df.nouns
        sentence_df, _, _ = pre_process_word_idx(sentence_df, v2idx, n2idx)
        sentence_df = fix_sentence_df(df, sentence_df)
    else:
        if args.sentence_dataframe == '':
            sentence_df = create_sentence_df(df, args.all_noun_classes)
        else:
            sentence_df = pd.read_pickle(args.sentence_dataframe)
    if not args.all_noun_classes:
        sentence_df.noun_class = sentence_df.noun_class.apply(lambda x: x[0])
        sentence_df['noun'] = sentence_df.nouns.apply(lambda x: x[0])
    else:
        sentence_df['noun'] = sentence_df.nouns

    type_mod = 'class' if args.class_ else 'uid'

    if args.caption:
        caption_rel_dict = create_PoS_rel_dict(df, sentence_df, type_='action_{}'.format(type_mod), all_nouns=args.all_noun_classes)
        save_output_dict(caption_rel_dict, args.out_dir, dataframe_name,
                         'caption' if not args.all_noun_classes else 'caption_allnouns')
    if args.verb:
        verb_rel_dict = create_PoS_rel_dict(df, sentence_df, type_='verb_{}'.format(type_mod))
        save_output_dict(verb_rel_dict, args.out_dir, dataframe_name, 'verb')
    if args.noun:
        noun_rel_dict = create_PoS_rel_dict(df, sentence_df, type_='noun_{}'.format(type_mod), all_nouns=args.all_noun_classes)
        save_output_dict(noun_rel_dict, args.out_dir, dataframe_name,
                         'noun' if not args.all_noun_classes else 'noun_allnouns')
 

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser('Creates relational feature files for MMEN/JPoSE')
    parser.add_argument('dataframe', type=str, help='Annotations dataframe to create feature files from')
    parser.add_argument('--sentence-dataframe', type=str, help='Sentence dataframe to use. ['']')
    parser.add_argument('--out-dir', type=str, help='Output directory to save files to. [./]')
    parser.add_argument('--caption', action='store_true', help='Create caption relational file. [True]')
    parser.add_argument('--no-caption', action='store_false', dest='caption', help='Do not create caption relational file.')
    parser.add_argument('--verb', action='store_true', help='Create verb relational file. [True]')
    parser.add_argument('--no-verb', action='store_false', dest='verb', help='Do not create verb relational file.')
    parser.add_argument('--noun', action='store_true', help='Create noun relational file. [True]')
    parser.add_argument('--no-noun', action='store_false', dest='noun', help='Do not noun caption relational file.')
    parser.add_argument('--class', action='store_true', dest='class_', help='Use classes when creating relational files. [True]')
    parser.add_argument('--no-class', action='store_false', dest='class_', help='Do not use classes when creating relational files.')
    parser.add_argument('--all-noun-classes', action='store_true', help='Use all noun classes when creating relational files.')

    parser.set_defaults(
            sentence_dataframe='',
            out_dir='./',
            caption=True,
            verb=True,
            noun=True,
            class_=True
    )
    main(parser.parse_args())
