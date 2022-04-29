import os
import sys
import argparse

import numpy as np
import pandas as pd
import torch as th
import torch.optim as optim
import torch.nn as nn

from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

import utils
import defaults.EPIC_JPOSE as EPIC_JPOSE

from models.jpose import JPOSE
from datasets.jpose_dataset import create_epic_jpose_dataset, create_youcook_jpose_dataset
from evaluation import nDCG
from evaluation import mAP
from train.train_mmen_triplet import initialise_nDCG_values
from train.train_jpose_triplet import initialise_jpose_nDCG_values, create_modality_dicts



def get_features(model, dataset, PoS_list, gpu=False, use_learned_comb_func=True):
    model.eval()

    vis_feat, txt_feat = dataset.get_eval_batch(PoS_list, gpu=gpu)
    #print("vis", ["{}: {}".format(k, v.shape) for k, v in vis_feat.items()])

    if use_learned_comb_func:
        comb_func = None
    else:
        comb_func = th.cat
    out_dict = model.forward({PoS: [{'v': vis_feat[PoS]}, {'t': txt_feat[PoS]}] for PoS in PoS_list if PoS != 'action'}, action_output=True,comb_func=comb_func)

    vis_out = out_dict[0]['v']
    txt_out = out_dict[1]['t']

    if gpu:
        vis_out = vis_out.cpu()
        txt_out = txt_out.cpu()
    vis_out = vis_out.detach().numpy()
    txt_out = txt_out.detach().numpy()

    return vis_out, txt_out



def eval_q2m(scores, q2m_gts):
    '''
    Image -> Text / Text -> Image
    Args:
      scores: (n_query, n_memory) matrix of similarity scores
      q2m_gts: list, each item is the positive memory ids of the query id
    Returns:
      scores: (recall@1, 5, 10, median rank, mean rank)
      gt_ranks: the best ranking of ground-truth memories
    '''
    n_q, n_m = scores.shape
    gt_ranks = np.zeros((n_q,), np.int32)

    for i in range(n_q):
        s = scores[i]
        sorted_idxs = np.argsort(-s)

        rank = n_m
        for k in q2m_gts[i]:
            tmp = np.where(sorted_idxs == k)[0][0]
            if tmp < rank:
                rank = tmp
        gt_ranks[i] = rank

    # compute metrics
    r1 = 100 * len(np.where(gt_ranks < 1)[0]) / n_q
    r5 = 100 * len(np.where(gt_ranks < 5)[0]) / n_q
    r10 = 100 * len(np.where(gt_ranks < 10)[0]) / n_q
    medr = np.median(gt_ranks) + 1
    meanr = gt_ranks.mean() + 1

    return r1, r5, r10, medr, meanr

def calculate_metrics(scores, i2t_gts, t2i_gts):
    import collections
    # caption retrieval
    cr1, cr5, cr10, cmedr, cmeanr = eval_q2m(scores, i2t_gts)
    # image retrieval
    ir1, ir5, ir10, imedr, imeanr = eval_q2m(scores.T, t2i_gts)
    # sum of recalls to be used for early stopping
    rsum = cr1 + cr5 + cr10 + ir1 + ir5 + ir10

    metrics = collections.OrderedDict()
    metrics['ir1'] = ir1
    metrics['ir5'] = ir5
    metrics['ir10'] = ir10
    metrics['imedr'] = imedr
    metrics['imeanr'] = imeanr
    metrics['cr1'] = cr1
    metrics['cr5'] = cr5
    metrics['cr10'] = cr10
    metrics['cmedr'] = cmedr
    metrics['cmeanr'] = cmeanr
    metrics['rsum'] = rsum

    return metrics


def test_epoch(model, dataset, PoS_list, gpu=False, use_learned_comb_func=True, dcg_ind=False, dataset_name=""):
    model.eval()

    vis_out, txt_out = get_features(model, dataset, PoS_list, gpu=False, use_learned_comb_func=True)
    import pickle

    vis_sim_matrix = vis_out.dot(txt_out.T)
    txt_sim_matrix = vis_sim_matrix.T

    if dataset_name == "epic100":
        i2t_gts = pickle.load(open("./data/relevancy/i2t_gts.pkl", "rb"))

    t2i_gts = {}
    # cap_num = 1
    for i, t_gts in enumerate(i2t_gts):
        for t_gt in t_gts:
            # if cap_num >= scores.shape[1]:
            #   break
            # else:
            #   cap_num += 1
            t2i_gts.setdefault(t_gt, [])
            t2i_gts[t_gt].append(i)
    # print(scores.shape, len(t2i_gts.keys()))
    #print(len(i2t_gts), len(t2i_gts))

    """pickle.dump(dataset['action'].IDCG['v'], open("IDCG_v.pkl", "wb"))
    pickle.dump(dataset['action'].IDCG['t'], open("IDCG_t.pkl", "wb"))
    pickle.dump(dataset['action'].relevancy_matrix, open("rel_mat.pkl", "wb"))"""

    if dataset_name == "epic100":
        vis_nDCG = nDCG.calculate_nDCG(vis_sim_matrix,
                dataset['action'].relevancy_matrix, dataset['action'].k_values['v'],
                IDCG=dataset['action'].IDCG['v'])
        txt_nDCG = nDCG.calculate_nDCG(txt_sim_matrix,
                dataset['action'].relevancy_matrix.T, dataset['action'].k_values['t'],
                IDCG=dataset['action'].IDCG['t'])
    elif dataset_name == "youcook2":
        rel_mat = dataset['action'].relevancy_matrix
        vis_k_counts = nDCG.calculate_k_counts(rel_mat)
        txt_k_counts = nDCG.calculate_k_counts(rel_mat.T)

        idcg_v = nDCG.calculate_IDCG(rel_mat, vis_k_counts)
        idcg_t = nDCG.calculate_IDCG(rel_mat.T, txt_k_counts)

        vis_nDCG = nDCG.calculate_nDCG(vis_sim_matrix, rel_mat, vis_k_counts, IDCG=idcg_v)
        txt_nDCG = nDCG.calculate_nDCG(vis_sim_matrix.T, rel_mat.T, txt_k_counts, IDCG=idcg_t)
    else:
        assert False, dataset_name
    print('nDCG: VT:{:.3f} TV:{:.3f} AVG:{:.3f}'.format(vis_nDCG, txt_nDCG, (vis_nDCG + txt_nDCG) / 2))
    if dataset_name == "epic100":
        vis_mAP = mAP.calculate_mAP(vis_sim_matrix,
                dataset['action'].relevancy_matrix)
        txt_mAP = mAP.calculate_mAP(txt_sim_matrix,
                dataset['action'].relevancy_matrix.T)
    elif dataset_name == "youcook2":
        vis_mAP = mAP.calculate_mAP(vis_sim_matrix, rel_mat)
        txt_mAP = mAP.calculate_mAP(vis_sim_matrix.T, rel_mat.T)
    else:
        assert False, dataset_name
    print('mAP: VT:{:.3f} TV:{:.3f} AVG:{:.3f}'.format(vis_mAP, txt_mAP, (vis_mAP + txt_mAP) / 2))

    if dataset_name == "epic100":
        metrics = calculate_metrics(vis_sim_matrix, i2t_gts, t2i_gts)
        print(f"{metrics['ir1']:.2f}; {metrics['ir10']:.2f}; {metrics['cr1']:.2f}; {metrics['cr10']:.2f}; "
              f"{vis_nDCG:.3f}; {txt_nDCG:.3f}; {((vis_nDCG + txt_nDCG) / 2):.3f}; "
              f"{vis_mAP:.3f}; {txt_mAP:.3f}; {((vis_mAP + txt_mAP) / 2):.3f}".replace(".", ","))

    else:
        print(f"{vis_nDCG:.3f}; {txt_nDCG:.3f}; {((vis_nDCG + txt_nDCG) / 2):.3f}; "
              f"{vis_mAP:.3f}; {txt_mAP:.3f}; {((vis_mAP + txt_mAP) / 2):.3f}".replace(".", ","))


def main(args):
    arg_file_path = args.MODEL_PATH.rsplit('/', 2)[0] + '/args.txt'
    model_args = utils.output.load_args(arg_file_path)

    print(model_args)
    if args.dataset == "epic100":
        dataset_fn = create_epic_jpose_dataset
    elif args.dataset == "youcook2":
        dataset_fn = create_youcook_jpose_dataset
    else:
        exit()

    if not hasattr(model_args, 'num_triplets'):
        setattr(model_args, 'num_triplets', 0)
    test_ds = dataset_fn(is_train=False, is_test=True, batch_size=model_args.batch_size, num_triplets=model_args.num_triplets,
                                        rgb_only=args.rgb, rgb_flow_only=args.rgb_flow)
    test_ds = initialise_jpose_nDCG_values(test_ds)

    modality_dicts, comb_func = create_modality_dicts(model_args, test_ds.x_size, test_ds.y_size)

    PoS_list = ['verb', 'noun', 'action']

    jpose = JPOSE(modality_dicts, comb_func=comb_func)
    jpose.load_state_dict(th.load(args.MODEL_PATH))

    if args.gpu:
        jpose.cuda()

    test_epoch(jpose, test_ds, PoS_list, gpu=args.gpu, use_learned_comb_func=args.comb_func, dcg_ind=args.dcg_ind, dataset_name=args.dataset)
    if args.challenge_submission != '':
        test_ds = create_epic_jpose_dataset(is_train=False, batch_size=model_args.batch_size, num_triplets=model_args.num_triplets, is_test=True)
        test_df = pd.read_pickle('./data/dataframes/EPIC_100_retrieval_test.pkl')
        test_sentence_df = pd.read_pickle('./data/dataframes/EPIC_100_retrieval_test_sentence.pkl')
        test_vis, test_txt = get_features(jpose, test_ds, PoS_list, gpu=args.gpu, use_learned_comb_func=args.comb_func)
        sim_mat = test_vis.dot(test_txt.T)
        out_dict = {}
        out_dict['version'] = 0.1
        out_dict['challenge'] = 'multi_instance_retrieval'
        out_dict['sim_mat'] = sim_mat
        out_dict['vis_ids'] = test_df.index
        out_dict['txt_ids'] = test_sentence_df.index
        out_dict['sls_pt'] = 2
        out_dict['sls_tl'] = 3
        out_dict['sls_td'] = 3
        pd.to_pickle(out_dict, args.challenge_submission, protocol=4)

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Test Joint Part-of-Speech Embedding Network (JPoSE) using Triplets")

    parser.add_argument('MODEL_PATH', type=str, help='Path of model to load')
    parser.add_argument('--gpu', type=bool, help='Whether or not to use the gpu for testin. [False]')
    parser.add_argument('--comb-func', type=bool, help='Whether or not to use the combination func for testing. [False]')
    parser.add_argument('--challenge-submission', type=str, help='Whether or not to create a challenge submission with given output path. ['']')
    parser.add_argument('--rgb', action='store_true')
    parser.add_argument('--rgb-flow', action='store_true')
    parser.add_argument('--dcg_ind', action='store_true')
    parser.add_argument('--dataset', default='epic100', choices=['epic100', 'youcook2'])

    parser.set_defaults(
            gpu=False,
            comb_func=False,
            challenge_submission=''
    )

    main(parser.parse_args())
