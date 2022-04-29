import pandas
import argparse

def main(args):
    synsets = pandas.read_pickle(args.synsets_file)
    print(synsets.head())
    out_df = synsets.copy()
    out_df.rename({'parsed_noun_classes': 'all_noun_classes',
                   'parsed_verb_classes': 'all_verb_classes'},
                  inplace=True)



if __name__ == "__main__":
    parse = argparse.ArgumentParser()
    parse.add_argument("synsets_file")
    parse.add_argument("out_dataframe_file")
    args = parse.parse_args()

    main(args)