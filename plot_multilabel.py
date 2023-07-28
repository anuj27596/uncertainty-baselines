import os
import sys
import time
import tabulate
import itertools
import multiprocessing

import numpy as np
import scipy.ndimage as ndimage
import sklearn.metrics as skm

import matplotlib.pyplot as plt

from tqdm import tqdm
# import tensorflow as tf

MAXREF = 95
REFRANGE = np.linspace(0, MAXREF / 100, MAXREF + 1)
NUM_CLASSES = 7

def tqdm_fake(it, *args, **kwargs):
    return it


def ecdf(x):
    return (x.argsort().argsort() + 1) / (x.size + 1)


# def split_referral_order(p):
#     c = np.zeros_like(p)
#     c[p >= 0.5] = ecdf(p[p >= 0.5])
#     c[p < 0.5] = 1 - ecdf(p[p < 0.5])
#     return c.argsort()

def split_referral_order(y_pred, y_true):

    y_hat = np.argmax(y_pred, axis=1)
    print(f"----- Debug ------ {y_pred.shape} {y_true.shape} {y_hat.shape}")

    order = {}
    for c in range(NUM_CLASSES):
        order[c] = {}
        ids = np.argsort(y_pred[y_hat == c], axis=0)[:, c] # sort by axis=c and take rankindex
        order[c]["y_pred"] = y_pred[y_hat == c][ids] # n_c x c
        order[c]["y_true"] = y_true[y_hat == c][ids] # n_c x c
    return order

def select_referral(order, r):
    '''
    returns: y_true_select, y_hat_select, y_pred_select
    '''
    y_true_select = []
    y_pred_select = []
    y_hat_select = []
    for c in range(NUM_CLASSES):
        n = order[c]["y_pred"].shape[0]
        n_remove = int(n*r)
        y_true_select.append(order[c]["y_true"][n_remove:])
        y_pred_select.append(order[c]["y_pred"][n_remove:])
        y_hat_select.append(np.repeat(c, n - n_remove))

    y_true_select = np.concatenate(y_true_select) # nxc
    y_pred_select = np.concatenate(y_pred_select) # nxc
    y_hat_select = np.concatenate(y_hat_select) # nx1

    return y_true_select, y_hat_select, y_pred_select
        
        

def process_func(m, d, seed, SR):
    '''
    m: model
    d: dataset (ood, id)
    '''
    # semaphore.acquire()
    start_time = time.time()

    if m.startswith('rb'):
        resdir = os.path.join(model_path[m], d, f'eval_results_{seed}')
        y = np.load(os.path.join(resdir, 'y_true.npy'))
        l = np.load(os.path.join(resdir, 'y_logit.npy'))
        p = 1 / (1 + np.exp(- l / 10))
    elif m.startswith('vit'):
        resdir = os.path.join(model_path[m], str(seed), d, 'eval_results_1000')
        print(resdir)
        y = np.load(os.path.join(resdir, 'y_true.npy'))[..., None].reshape(-1, NUM_CLASSES) # n X c
        if m == 'vit_osp':
            l = np.load(os.path.join(resdir, 'logits.npy')).reshape(-1, 2) # n x c
            pred_class = l.argmax(axis = 1) # n x 1
            pred_certainty = l.max(axis = 1) # n x 1
            p = 0.5 + (pred_class - 0.5) * ecdf(pred_certainty) 
            p = p[..., None]
        else:
            p = np.load(os.path.join(resdir, 'y_pred.npy'))[..., None].reshape(-1, NUM_CLASSES) # n x c
        # e = np.load(os.path.join(resdir, 'pre_logits.npy'))
        # e = e.reshape(-1, e.shape[-1])

    metrics = {
        score: np.zeros((REFRANGE.size,1)) # c x r
        for score in ["Accuracy", "AvgPrec"]
    }
    
    if SR:
        order_split = split_referral_order(p, y)
    else:
        order =  np.max(p, axis=1).argsort()
        
    for r_idx, r in enumerate(REFRANGE):

        if SR:
            y_true_selected, y_hat_selected, y_pred_selected = select_referral(order_split, r)
        else:
            ids = order[int(r*len(order)):]
            y_true_selected, y_hat_selected, y_pred_selected = y[ids], np.argmax(p, axis=1)[ids].reshape(-1,), p[ids] 
            
        metrics['Accuracy'][r_idx] = skm.accuracy_score(np.argmax(y_true_selected, axis=1), y_hat_selected)
        metrics['AvgPrec'][r_idx] = skm.average_precision_score(y_true_selected, y_pred_selected, average="weighted")
            

        print(f"----------- debug ------- y_true_selected: {y_true_selected.shape} y_hat_selected: {y_hat_selected.shape}")

 
        # if y[ids, class_id].mean() not in (0, 1):
        #     metrics['AUROC'][class_id, r_idx] = skm.roc_auc_score(
        #         y[ids, class_id], p[ids, class_id])
        #     metrics['AvgPrec'][class_id, r_idx] = skm.average_precision_score(
        #         y[ids, class_id], p[ids, class_id])
        # else:
        #     metrics['AUROC'][class_id, r_idx:] = metrics['AUROC'][class_id, r_idx - 1]
        #     metrics['AvgPrec'][class_id, r_idx:] = metrics['AvgPrec'][class_id, r_idx - 1]
        #     break

    # print(f"********** MTERICS = {metrics} ***************")
    for score in metrics:
        np.save(f'dump/{m}-{d}-{seed}-{SR}-{score}.npy', metrics[score])

    print(f'[{time.time() - start_time:.2f}] done: {m}/{d}/{seed}{"/SR" if SR else ""}')
    # semaphore.release()
    return


model_name = {
    'vit': 'ViT',
    'vit_dan': 'DAN',
    'vit_dan_ens': 'DAN(Ens)',
    'vit_simclr_joint': 'SimCLR(joint)',
    'vit_simclr_joint_ood': 'SimCLR-OOD(joint)',
    'vit_simclr_seq': 'SimCLR(seq)',
    'vit_simclr_seq_ood': 'SimCLR-OOD(seq)',
    'vit_mim_seq': 'MIM(seq)',
    'vit_mim_seq_ood': 'MIM-OOD(seq)',
    'vit_osp': 'OSP',
    # 'vit_local_seq': 'LocalCL(seq)',
    # 'rb_deterministic': 'ResNet50',
    # 'rb_dropout': 'ResNet50+MCD',
}
model_path = {
    'vit': 'eval_results/vit',
    'vit_dan': 'eval_results/in21k_dan/grl_1/layers_2/dim_256/loss_1',
    'vit_dan_ens': 'eval_results/in21k_dan_ens',
    'vit_simclr_joint': 'eval_results/in21k_simclr_joint_auroc',
    'vit_simclr_joint_ood': 'eval_results/in21k_simclr_joint_ood',
    'vit_simclr_seq': 'eval_results/in21k_simclr_seq',
    'vit_simclr_seq_ood': 'eval_results/in21k_simclr_seq_ood',
    'vit_mim_seq': 'eval_results/in21k_mim_seq_0.6',
    'vit_mim_seq_ood': 'eval_results/in21k_mim_ood_seq_0.6',
    'vit_osp': 'eval_results/in21k_osp',
    # 'vit_local_seq': 'eval_results/in21k_local_spatial_seq',
    # 'rb_deterministic': '/troy/anuj/gub-mod/results/ub-results-withemb/aptos/deterministic_k1_indomain_mc5/single',
    # 'rb_dropout': '/troy/anuj/gub-mod/results/ub-results-withlogit/aptos/dropout_k1_indomain_mc5/single',
}
plot_color = {
    'vit': 'k',
    'vit_simclr_seq_ood': 'b',
    'vit_simclr_joint_ood': 'c',
    'vit_dan_ens': 'c',
    'vit_dan': 'c',
    'vit_simclr_seq': 'b',
    'vit_simclr_joint': 'c',
    'vit_mim_seq': 'r',
    'vit_mim_seq_ood': 'r',
    'vit_osp': 'y',
    # 'vit_local_seq': 'y',
    # 'rb_deterministic': 'k--',
    # 'rb_dropout': 'y--',
}
seeds = {
    'vit_local_seq': [1],
    'rb_dropout': range(3),
    'vit_simclr_joint': [*range(5), 69],
    'vit': [1]
}





if __name__ == '__main__':
    

    models = [
        'vit',
        # 'vit_dan',
        # 'vit_simclr_seq',
        # 'vit_simclr_seq_ood',
        # 'vit_simclr_joint',
        # 'vit_simclr_joint_ood',
        # 'vit_dan_ens',
        # 'vit_mim_seq',
        # 'vit_mim_seq',
        # 'vit_osp',
        # 'vit_local_seq',
        # 'rb_deterministic',
        # 'rb_dropout',
    ]
    datasets = [
        'in_domain_test',
        'in_domain_validation',
        'ood_test',
        'ood_validation',
    ]
    

    metrics = dict(
        Accuracy = {},
        # AUROC = {},
        AvgPrec = {},
    )

    proc_obj_list = []
    # semaphore = multiprocessing.Semaphore(value = 8)

    os.makedirs('dump', exist_ok = True)


    start_time = time.time()

    for m, d, SR in itertools.product(models, datasets, (False, True)):
        for seed in seeds.get(m, range(6)):
            if os.path.exists(f'dump/{m}-{d}-{seed}-{SR}-AvgPrec.npy'):
                continue

            process_func(m, d, seed, SR)

    print('-' * 79, file = sys.stderr)
    print(f'[{time.time() - start_time:.2f}] done all', file = sys.stderr)


    for m, d, SR, score in tqdm(list(itertools.product(models, datasets, (False, True), metrics.keys()))):
        metrics[score][m, d, SR] = np.stack([
            np.load(f'dump/{m}-{d}-{seed}-{SR}-{score}.npy')
            for seed in seeds.get(m, range(6))
        ])

    for d in datasets:
        plt.figure(figsize = (5 * len(metrics), 5))

        records = []

        for idx1, p in enumerate(metrics):
            plt.subplot(1, len(metrics), idx1 + 1)
            plt.grid(visible = True, linestyle = '--', alpha = 0.5)
            
            for idx2, (SR, m) in enumerate(itertools.product((False, True), models)):
            # for SR, m in itertools.product((False, ), models):
                # if m not in ('vit',) and SR:
                #     continue
                mean = metrics[p][m, d, SR].reshape(-1,)
                # import pdb; pdb.set_trace()
                plt.plot(REFRANGE, mean, plot_color[m], linestyle = ('--' if SR else '-'))
                
                if idx1 == 0:
                    records.append([
                        model_name[m] + (" [SR]" if SR else ""),
                        *([''] * (len(metrics) * 2))
                    ])
                records[idx2][idx1 * 2 + 1] = mean[0].round(4)
                records[idx2][idx1 * 2 + 2] = mean.mean().round(4)

            plt.legend(
                [model_name[m] for m in models]
                + [model_name[m] + ' [SR]' for m in models]
            )

            # for SR, m in itertools.product((False, True), models):
            #     if m != 'vit' and SR:
            #         continue
            #     mean = metrics[p][m, d, SR].reshape(-1,)
            #     std = np.std(metrics[p][m, d, SR].reshape(-1,)) # ACROSS THE SEED
            #     plt.fill_between(REFRANGE, mean - std, mean + std, color = plot_color[m], alpha = 0.05)

            plt.xlabel('Referral Rate')
            plt.ylabel(p)

        print('\n' + d)
        headers = ['Model', *[prefix + metric for metric in metrics for prefix in ('', 'AUPCC-')]]
        print(tabulate.tabulate(records, headers, tablefmt = 'outline'))

        plt.suptitle(d)
        plt.tight_layout()

        plt.savefig(f'plots/ref-{d}.png')

    plt.show()
