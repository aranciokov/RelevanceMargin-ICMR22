import os
import sys

import numpy as np
import torch as th
import torch.optim as optim
import torch.nn as nn

from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

import utils
import parsing
import defaults.EPIC_JPOSE as EPIC_JPOSE

from models.jpose import JPOSE
from datasets.jpose_dataset import create_epic_jpose_dataset, create_youcook_jpose_dataset
from losses.triplet import TripletLoss
from evaluation import nDCG
from train.train_mmen_triplet import initialise_nDCG_values



def relevance(v_i, v_j, N_i, N_j):
    verb_iou = len(set(v_i).intersection(set(v_j))) / len(set(v_i).union(set(v_j)))
    noun_iou = len(set(N_i).intersection(set(N_j))) / len(set(N_i).union(set(N_j)))
    return 0.5 * (verb_iou + noun_iou)


def get_relevances(x_classes, y_classes):
    x_verbs = [x[0] if isinstance(x[0], list) else [x[0]] for x in x_classes]
    x_nouns = [x[1] if isinstance(x[1], list) else [x[1]] for x in x_classes]
    y_verbs = [x[0] if isinstance(x[0], list) else [x[0]] for x in y_classes]
    y_nouns = [x[1] if isinstance(x[1], list) else [x[1]] for x in y_classes]
    rel_mat = th.tensor([relevance(v1, v2, n1, n2) for v1, n1, v2, n2 in zip(x_verbs, x_nouns, y_verbs, y_nouns)])

    return rel_mat

def get_relevance_verb(x_verbs, y_verbs):
    rel_mat = th.tensor([relevance(v1, v2, [0], [0]) if isinstance(v1, list) \
                             else relevance([v1], [v2], [0], [0]) for v1, v2 in zip(x_verbs, y_verbs)])
    return rel_mat


def get_relevance_noun(x_nouns, y_nouns):
    rel_mat = th.tensor([relevance([0], [0], n1, n2) for n1, n2 in zip(x_nouns, y_nouns)])
    return rel_mat

def parse_classes(act_classes):
    # act_classes: List[str] where str->"{verb_cls}_{noun_cls}" or "{verb_cls}_{noun_cls_1,noun_cls_2,...}"
    """tmp = []
    for act_cls in act_classes:
        print(act_cls)
        tmp.append((int(act_cls.split("_")[0].split(",")[0])
             if len(act_cls.split("_")[0].split(",")) < 2
             else list(map(int, act_cls.split("_")[0].split(",")))
             ,
             int(act_cls.split("_")[1].split(",")[0])
             if len(act_cls.split("_")[1].split(",")) < 2
             else list(map(int, act_cls.split("_")[1].split(",")))
             ))
    return tmp"""

    try:
        return [(int(act_cls.split("_")[0].split(",")[0])
             if len(act_cls.split("_")[0].split(",")) < 2
             else list(map(int, act_cls.split("_")[0].split(",")))
             ,
             int(act_cls.split("_")[1].split(",")[0])
             if len(act_cls.split("_")[1].split(",")) < 2
             else list(map(int, act_cls.split("_")[1].split(",")))
             ) for act_cls in act_classes]
    except:
        tmp = []
        for act_cls in act_classes:
            print(act_cls)
            tmp.append((int(act_cls.split("_")[0].split(",")[0])
                 if len(act_cls.split("_")[0].split(",")) < 2
                 else list(map(int, act_cls.split("_")[0].split(",")))
                 ,
                 int(act_cls.split("_")[1].split(",")[0])
                 if len(act_cls.split("_")[1].split(",")) < 2
                 else list(map(int, act_cls.split("_")[1].split(",")))
                 ))
        return tmp

"""def parse_classes(act_classes):
    # act_classes: List[str] where str->"{verb_cls}_{noun_cls}" or "{verb_cls}_{noun_cls_1,noun_cls_2,...}"
    return [(int(act_cls.split("_")[0]),
             int(act_cls.split("_")[1].split(",")[0])
             if len(act_cls.split("_")[1].split(",")) < 2
             else list(map(int, act_cls.split("_")[1].split(",")))
             ) for act_cls in act_classes]"""


def sample_triplets_for_dataset(dataset, PoS_list, cross_modal_weights_dict, sampling_method='random'):
    for cross_modal_pair in cross_modal_weights_dict:
        if cross_modal_weights_dict[cross_modal_pair] > 0:
            dataset.sample_triplets(PoS_list, cross_modal_pair, sampling_method=sampling_method)


def initialise_jpose_nDCG_values(dataset):
    verb_IDCG, verb_k_values = initialise_nDCG_values(dataset.verb)
    dataset.verb.IDCG = verb_IDCG
    dataset.verb.k_values = verb_k_values

    noun_IDCG, noun_k_values = initialise_nDCG_values(dataset.noun)
    dataset.noun.IDCG = noun_IDCG
    dataset.noun.k_values = noun_k_values

    action_IDCG, action_k_values = initialise_nDCG_values(dataset.action)
    dataset.action.IDCG = action_IDCG
    dataset.action.k_values = action_k_values

    return dataset


def create_loss(args, pos_weight=1.0):
    loss_func = TripletLoss
    tt_loss = loss_func(args.margin, args.tt_weight * pos_weight)
    tv_loss = loss_func(args.margin, args.tv_weight * pos_weight)
    vt_loss = loss_func(args.margin, args.vt_weight * pos_weight)
    vv_loss = loss_func(args.margin, args.vv_weight * pos_weight)
    loss_dict = {'tt': tt_loss, 'tv': tv_loss, 'vt': vt_loss, 'vv': vv_loss}
    return loss_dict


def create_triplet_losses(args, PoS_list):
    loss_dict = {}
    if 'verb' in PoS_list:
        loss_dict['verb'] = create_loss(args, args.verb_weight)
    if 'noun' in PoS_list:
        loss_dict['noun'] = create_loss(args, args.noun_weight)
    if 'action' in PoS_list:
        loss_dict['action'] = create_loss(args, args.action_weight)
    return loss_dict


def train_epoch(model, dataset, PoS_list, cross_modal_weights_dict, loss_dict,
        optimiser, writer, epoch_num, gpu=False, use_learned_comb_func=True,
        sampling_method='random', rel_margin=False, use_cosine=False, args=None):
    model.train()
    accum_loss = 0.
    avail_triplets = [pair for pair in cross_modal_weights_dict if cross_modal_weights_dict[pair] > 0]
    num_batches = int(np.ceil(len(dataset) / dataset.batch_size))
    for i, batch_dict in enumerate(tqdm(dataset.get_triplet_batch(PoS_list, avail_triplets, gpu=gpu), total=num_batches)):
        optimiser.zero_grad()
        for PoS in PoS_list:

            if "v_t_cls" in batch_dict[PoS] and "t_v_cls" in batch_dict[PoS]:
                tv_anc_cls, tv_pos_cls, tv_neg_cls = batch_dict[PoS]["t_v_cls"]
                vt_anc_cls, vt_pos_cls, vt_neg_cls = batch_dict[PoS]["v_t_cls"]
                batch_dict[PoS].pop("t_v_cls")
                batch_dict[PoS].pop("v_t_cls")
                #print(PoS, "t_v", [(tv_anc_cls[i], tv_pos_cls[i], tv_neg_cls[i]) for i in range(5)],
                #      "v_t", [(vt_anc_cls[i], vt_pos_cls[i], vt_neg_cls[i]) for i in range(5)])
                assert "t_v_cls" not in batch_dict[PoS] and "v_t_cls" not in batch_dict[PoS]

            if "t_t_cls" in batch_dict[PoS] and "v_v_cls" in batch_dict[PoS]:
                tt_anc_cls, tt_pos_cls, tt_neg_cls = batch_dict[PoS]["t_t_cls"]
                vv_anc_cls, vv_pos_cls, vv_neg_cls = batch_dict[PoS]["v_v_cls"]
                batch_dict[PoS].pop("t_t_cls")
                batch_dict[PoS].pop("v_v_cls")
                #print(PoS, "t_t", [(tt_anc_cls[i], tt_pos_cls[i], tt_neg_cls[i]) for i in range(5)],
                #      "v_v", [(vv_anc_cls[i], vv_pos_cls[i], vv_neg_cls[i]) for i in range(5)])
                assert "t_t_cls" not in batch_dict[PoS] and "v_v_cls" not in batch_dict[PoS]

            for cross_modal_pair in batch_dict[PoS]:
                anc, pos_, neg = batch_dict[PoS][cross_modal_pair]
                anc_id = cross_modal_pair[0]
                other_id = cross_modal_pair[1]
                if PoS == 'action':
                    for sub_PoS in anc.keys():
                        forward_dict[sub_PoS] = [
                                {anc_id: anc[sub_PoS]},
                                {other_id: pos_[sub_PoS]},
                                {other_id: neg[sub_PoS]}
                        ]
                    if use_learned_comb_func:
                        out_dict = model.forward(forward_dict, action_output=True)
                    else:
                        out_dict = model.forward(forward_dict, action_output=True, comb_func=th.cat)
                else:
                    forward_dict = {PoS:
                            [
                                {anc_id: anc},
                                {other_id: pos_},
                                {other_id: neg}
                            ]
                    }
                    out_dict = model.forward(forward_dict)
                anc_loss = out_dict[0][anc_id]
                pos_loss = out_dict[1][other_id]
                neg_loss = out_dict[2][other_id]
                if rel_margin:
                    if cross_modal_pair == "tv":
                        _a, _p, _n = tv_anc_cls, tv_pos_cls, tv_neg_cls
                    elif cross_modal_pair == "vt":
                        _a, _p, _n = vt_anc_cls, vt_pos_cls, vt_neg_cls
                    elif cross_modal_pair == "tt":
                        _a, _p, _n = tt_anc_cls, tt_pos_cls, tt_neg_cls
                    elif cross_modal_pair == "vv":
                        _a, _p, _n = vv_anc_cls, vv_pos_cls, vv_neg_cls

                    if PoS == "action":
                        delta = 1 - \
                                get_relevances(parse_classes(_a), parse_classes(_n))
                    elif PoS == "verb":
                        delta = 1 - get_relevance_verb(_a, _n)
                    else:
                        delta = 1 - get_relevance_noun(_a, _n)

                    if gpu:
                        delta = delta.cuda()
                    loss = loss_dict[PoS][cross_modal_pair](anc_loss, pos_loss, neg_loss, margin=delta, cos_sim=use_cosine)
                else:
                    loss = loss_dict[PoS][cross_modal_pair](anc_loss, pos_loss, neg_loss, cos_sim=use_cosine)
                #print(PoS, cross_modal_pair, loss.item())
                loss = loss / (len(PoS_list) * len(batch_dict[PoS])) #TODO: count number of actual batches
                accum_loss += loss.data.item()
                loss.backward()
                if PoS == "action":
                    with th.no_grad():
                        d_pos = (anc_loss - pos_loss).pow(2).sum(1)
                        d_neg = (anc_loss - neg_loss).pow(2).sum(1)
                        import torch.nn.functional as F
                        _delta = delta if rel_margin else loss_dict[PoS][cross_modal_pair].margin
                        losses =  F.relu(_delta + d_pos - d_neg)
                        import numpy
                        h = len(losses[losses > 0]) / numpy.prod(losses.shape)
                    writer.add_scalar('hardness', h, epoch_num*num_batches+i)
        optimiser.step()
    print('...{}'.format(accum_loss))
    writer.add_scalar('loss', accum_loss, epoch_num)


def test_epoch(model, dataset, PoS_list, writer, epoch_num, out_dir, gpu=False, use_learned_comb_func=True, final_run=False):
    model.eval()

    vis_feat, txt_feat = dataset.get_eval_batch(PoS_list, gpu=gpu)

    for PoS in PoS_list:
        if PoS == 'action':
            continue
        out_dict = model.forward({PoS: [{'v': vis_feat[PoS]}, {'t': txt_feat[PoS]}]})

        vis_out = out_dict[0]['v']
        txt_out = out_dict[1]['t']

        if gpu:
            vis_out = vis_out.cpu()
            txt_out = txt_out.cpu()
        vis_out = vis_out.detach().numpy()
        txt_out = txt_out.detach().numpy()

        vis_sim_matrix = vis_out.dot(txt_out.T)
        txt_sim_matrix = vis_sim_matrix.T

        vis_nDCG = nDCG.calculate_nDCG(vis_sim_matrix,
                dataset[PoS].relevancy_matrix, dataset[PoS].k_values['v'],
                IDCG=dataset[PoS].IDCG['v'])
        txt_nDCG = nDCG.calculate_nDCG(txt_sim_matrix,
                dataset[PoS].relevancy_matrix.T, dataset[PoS].k_values['t'],
                IDCG=dataset[PoS].IDCG['t'])
        print('{}: {:.3f} {:.3f}'.format(PoS, vis_nDCG, txt_nDCG))
        writer.add_scalars('nDCG/', {'{}/vid2txt'.format(PoS): vis_nDCG,
                                     '{}/txt2vid'.format(PoS): txt_nDCG}, epoch_num)

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

    vis_sim_matrix = vis_out.dot(txt_out.T)
    txt_sim_matrix = vis_sim_matrix.T

    vis_nDCG = nDCG.calculate_nDCG(vis_sim_matrix,
            dataset['action'].relevancy_matrix, dataset['action'].k_values['v'],
            IDCG=dataset['action'].IDCG['v'])
    txt_nDCG = nDCG.calculate_nDCG(txt_sim_matrix,
            dataset['action'].relevancy_matrix.T, dataset['action'].k_values['t'],
            IDCG=dataset['action'].IDCG['t'])
    print('action: {:.3f} {:.3f}'.format(vis_nDCG, txt_nDCG))
    writer.add_scalars('nDCG/', {'action/vid2txt': vis_nDCG,
                                 'action/txt2vid': txt_nDCG}, epoch_num)
    if final_run:
        vis_txt_identifier = np.concatenate((np.ones(vis_out.shape[0]), np.zeros(txt_out.shape[0])))
        writer.add_embedding(np.concatenate((vis_out, txt_out)), metadata=vis_txt_identifier, global_step=epoch_num)
        all_vis_nDCG = nDCG.calculate_nDCG(vis_sim_matrix,
                dataset['action'].relevancy_matrix, dataset['action'].k_values['v'],
                IDCG=dataset['action'].IDCG['v'], reduction=None)
        all_txt_nDCG = nDCG.calculate_nDCG(txt_sim_matrix,
                dataset['action'].relevancy_matrix.T, dataset['action'].k_values['t'],
                IDCG=dataset['action'].IDCG['t'], reduction=None)
        x_idxs = [dataset.action.x_idx_to_uid[idx] for idx in dataset.action._x_idxs]
        y_idxs = dataset.action._y_idxs
        all_dict = {'vis2txt': all_vis_nDCG, 'txt2vis': all_txt_nDCG, 
                'x_idxs': x_idxs, 'y_idxs': y_idxs}
        utils.output.save_results(out_dir, all_dict, 'pre_mean_nDCG')

def create_modality_dicts(args, x_size, y_size):
    modality_dicts = {
        'verb':{
            't': {
                'num_layers': args.num_layers,
                'layer_sizes': [y_size, args.embedding_size]
            },
            'v': {
                'num_layers': args.num_layers,
                'layer_sizes': [x_size, args.embedding_size]
            }
        },
        'noun':{
            't': {
                'num_layers': args.num_layers,
                'layer_sizes': [y_size, args.embedding_size]
            },
            'v': {
                'num_layers': args.num_layers,
                'layer_sizes': [x_size, args.embedding_size]
            }
        }
    }

    if args.comb_func in ['sum', 'max', 'cat']:
        comb_func = {args.comb_func: []}
    elif args.comb_func == 'fc':
        comb_func = {args.comb_func: (2 * args.embedding_size, args.embedding_size)}
    else:
        raise NotImplementedError('Combined Function: {} not implemented.'.format(args.comb_func))

    return modality_dicts, comb_func

def main(args):
    print(args)

    if args.dataset == "epic100":
        dataset_fn = create_epic_jpose_dataset
    elif args.dataset == "youcook2":
        dataset_fn = create_youcook_jpose_dataset
    else:
        exit()

    train_ds = dataset_fn(is_train=True, batch_size=args.batch_size, num_triplets=args.num_triplets,
                                         all_nouns=args.all_noun_classes,
                                         rgb_only=args.rgb, rgb_flow_only=args.rgb_flow)
    test_ds = dataset_fn(is_train=False, batch_size=args.batch_size, num_triplets=args.num_triplets,
                                        rgb_only=args.rgb, rgb_flow_only=args.rgb_flow)

    full_out_dir = utils.output.get_out_dir(args)
    print('#Saving models and results in {}'.format(full_out_dir))

    writer = SummaryWriter(log_dir=os.path.join(full_out_dir, 'results'))

    modality_dicts, comb_func = create_modality_dicts(args, train_ds.x_size, train_ds.y_size)

    cross_modal_weights_dict = {'tt': args.tt_weight,
                                'tv': args.tv_weight,
                                'vt': args.vt_weight,
                                'vv': args.vv_weight
    }
    PoS_list = ['verb', 'noun', 'action']
    sampling_method = 'random'

    jpose = JPOSE(modality_dicts, comb_func=comb_func)

    loss_dict = create_triplet_losses(args, PoS_list)

    if args.gpu:
        jpose.cuda()
        for pos in loss_dict:
            for modality_pair in loss_dict[pos]:
                loss_dict[pos][modality_pair].cuda()

    optimiser = optim.SGD(jpose.parameters(), lr=args.learning_rate, momentum=args.momentum)


    test_ds = initialise_jpose_nDCG_values(test_ds)

    for epoch_num in range(args.num_epochs):
        print('Beginning Epoch {}'.format(epoch_num + 1))
        if (epoch_num + 1) % args.triplet_sampling_rate == 1:
            sample_triplets_for_dataset(train_ds, PoS_list, cross_modal_weights_dict, sampling_method)
        train_epoch(jpose, train_ds, PoS_list, cross_modal_weights_dict, loss_dict, optimiser, writer, epoch_num, gpu=args.gpu, use_learned_comb_func=epoch_num >= args.comb_func_start, sampling_method=sampling_method,
                    rel_margin=args.rel_margin, use_cosine=args.cosine, args=args)

        if (epoch_num + 1) % args.checkpoint_rate == 1:
            utils.output.save_model(full_out_dir, jpose, epoch_num)
        print((epoch_num+1)==args.num_epochs)
        test_epoch(jpose, test_ds, PoS_list, writer, epoch_num, full_out_dir, gpu=args.gpu, use_learned_comb_func=epoch_num >= args.comb_func_start)

    utils.output.save_model(full_out_dir, jpose, epoch_num)
    writer.flush()
    writer.close()
    print('#Saved models and results in {}'.format(full_out_dir))

if __name__ == '__main__':
    parser = parsing.get_JPoSE_parser("Joint Part-of-Speech Embedding Network (JPoSE) using Triplets")

    parser.add_argument('--num-triplets', type=int, help='How many triplets to sample per anchor. [{}]'.format(EPIC_JPOSE.num_triplets))
    parser.add_argument('--triplet-sampling-rate', type=int, help='Number of epochs in between triplet sampling. [{}]'.format(EPIC_JPOSE.triplet_sampling_rate))
    parser.add_argument('--rel-margin', action='store_true', help='Number of epochs in between triplet sampling. [{}]'.format(EPIC_JPOSE.triplet_sampling_rate))
    parser.add_argument('--all-noun-classes', action='store_true')
    parser.add_argument('--cosine', action='store_true')
    parser.add_argument('--rgb', action='store_true')
    parser.add_argument('--rgb-flow', action='store_true')
    parser.add_argument('--dataset', default='epic100', choices=['epic100', 'youcook2'])

    parser.set_defaults(
            num_triplets=EPIC_JPOSE.num_triplets,
            triplet_sampling_rate=EPIC_JPOSE.triplet_sampling_rate,
    )
    main(parser.parse_args())
