import os
import sys
import pickle
import pandas as pd
import numpy as np
#import word2vec as wv
from tqdm import tqdm

from scripts.create_multiVNs_feature_files import create_sentence_df, pre_process_word_idx, fix_sentence_df

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
    fn = create_sent_rel_dict if type_.startswith("action") else create_rel_dict
    vid_to_class_dict, class_to_vid_dict, vid_to_classes_dict = fn(df, type_ if type_.startswith("action") else f"parsed_{type_}es", all_nouns)
    sent_to_class_dict, class_to_sent_dict, sent_to_classes_dict = fn(sentence_df, type_, all_nouns)
    rel_dict['class2vid'] = class_to_vid_dict
    rel_dict['class2sent'] = class_to_sent_dict
    rel_dict["vid2class"] = vid_to_class_dict
    rel_dict["vid2classes"] = vid_to_classes_dict
    rel_dict["sent2class"] = sent_to_class_dict
    rel_dict["sent2classes"] = sent_to_classes_dict
    return rel_dict


def create_sent_rel_dict(df, column, all_nouns=False):
    class_to_mod_dict = {}
    mod_to_class_dict = {}
    mod_to_classes_dict = {}

    for i, row in tqdm(df.iterrows(), total=len(df)):
        all_cls = row[column]  # action_class -> '{V}_{N}', need to access parsed_{verb|noun}_classes to get all
        #print(all_cls)

        mod_to_class_dict[i] = all_cls
        if all_cls not in class_to_mod_dict:
            try:
                class_to_mod_dict[all_cls] = [i]
            except:
                import bpdb;
                bpdb.set_trace()
        else:
            class_to_mod_dict[all_cls].append(i)

        try:
            mod_to_classes_dict[i] = [list(row.parsed_verb_classes), list(row.parsed_noun_classes)]
        except:
            mod_to_classes_dict[i] = [list(row.verb_class), list(row.noun_class)]
        #print([list(row.parsed_verb_classes), list(row.parsed_noun_classes)])
        #input()

    return mod_to_class_dict, class_to_mod_dict, mod_to_classes_dict


def create_rel_dict(df, column, all_nouns=False):
    class_to_mod_dict = {}
    mod_to_class_dict = {}
    mod_to_classes_dict = {}

    for i, row in tqdm(df.iterrows(), total=len(df)):
        # want to use *all* the classes
        #print(row)
        all_cls = row[column]
        if isinstance(all_cls, list):
            mod_to_classes_dict[i] = all_cls
            mod_to_class_dict[i] = all_cls[0] if len(all_cls) > 0 else -1
            for c in all_cls:
                if c not in class_to_mod_dict:
                    try:
                        class_to_mod_dict[c] = [i]
                    except:
                        import bpdb; bpdb.set_trace()
                else:
                    class_to_mod_dict[c].append(i)
        else:
            mod_to_class_dict[i] = all_cls
            mod_to_classes_dict[i] = all_cls
            if all_cls not in class_to_mod_dict:
                try:
                    class_to_mod_dict[all_cls] = [i]
                except:
                    import bpdb;
                    bpdb.set_trace()
            else:
                class_to_mod_dict[all_cls].append(i)

    return mod_to_class_dict, class_to_mod_dict, mod_to_classes_dict


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
            sentence_df = create_sentence_df(df)
        else:
            sentence_df = pd.read_pickle(args.sentence_dataframe)

    print(sentence_df)
    # multiple verb classes and multiple noun classes
    type_mod = 'class' if args.class_ else 'uid'

    if args.caption:
        caption_rel_dict = create_PoS_rel_dict(df, sentence_df, type_='action_{}'.format(type_mod), all_nouns=True)
        save_output_dict(caption_rel_dict, args.out_dir, dataframe_name, 'caption_allnouns')
    exit()
    if args.verb:
        verb_rel_dict = create_PoS_rel_dict(df, sentence_df, type_='verb_{}'.format(type_mod), all_nouns=True)
        save_output_dict(verb_rel_dict, args.out_dir, dataframe_name, 'verb_allnouns')
    if args.noun:
        noun_rel_dict = create_PoS_rel_dict(df, sentence_df, type_='noun_{}'.format(type_mod), all_nouns=True)
        save_output_dict(noun_rel_dict, args.out_dir, dataframe_name, 'noun_allnouns')
 

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

    parser.set_defaults(
            sentence_dataframe='',
            out_dir='./',
            caption=True,
            verb=True,
            noun=True,
            class_=True
    )
    main(parser.parse_args())
