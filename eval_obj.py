from source.layers.common_fns import positionalencoding2d
from typing import Dict, Callable
from collections import OrderedDict
import sys
import os

# sys.path.append(rootdir)
import torch
import torch.nn as nn
import torch.optim
import tqdm
from ema_pytorch import EMA
import matplotlib.pyplot as plt

from source.models.objs.knet import AKOrN
from source.models.objs.vit import ViT
from source.utils import get_worker_init_fn
from torch.nn import functional as F
from source.layers.common_layers import RGBNormalize
import numpy as np

import timm
from timm.models import VisionTransformer
from source.utils import gen_saccade_imgs, apply_pca_torch, str2bool
import argparse

from source.evals.objs.mbo import calc_mean_best_overlap
from source.evals.objs.fgari import calc_fgari_score

from typing import Callable


class Wrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.module = model


noise = 0.0

# Using CPU for PCA can be very slow depending on your CPU hardware.
USE_GPU_FOR_PCA = True


def remove_all_forward_hooks(model: torch.nn.Module) -> None:
    for name, child in model._modules.items():
        if child is not None:
            if hasattr(child, "_forward_hooks"):
                child._forward_hooks: Dict[int, Callable] = OrderedDict()
            remove_all_forward_hooks(child)


def model_preds(model, org_images):
    activation = {}
    imsize_h, imsize_w = org_images.shape[-2], org_images.shape[-1]

    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()

        return hook

    if isinstance(model, AKOrN):
        model.out[0].register_forward_hook(get_activation("z"))
    elif isinstance(model, ViT):
        model.out[0].register_forward_hook(get_activation("z"))

    else:
        raise Exception()

    model.eval()
    imgs = org_images.cuda()

    with torch.no_grad():
        if (
            isinstance(model, AKOrN)
            or isinstance(model, ViT)
        ):
            output, _xs = model(imgs, return_xs=True)
        else:
            output = model(imgs)
            _xs = None
    v = activation["z"]

    if isinstance(model, AKOrN) or isinstance(model, ViT):
        v = F.normalize(v, dim=1)
    # elif isinstance(model, ViTWrapper):
    #    v = F.normalize(v, dim=2)
    #    v = v.permute(0, 2, 1)[..., 1:]
    #    h, w = int(np.sqrt(x.shape[-1])), int(np.sqrt(x.shape[-1]))  # estimated inpsize
    #    v = v.unflatten(-1, (h, w))
    remove_all_forward_hooks(model)
    return v


def clustering(x, h, w, method="spectral", n_clusters=3):
    from sklearn.cluster import KMeans

    if method == "agglomerative":
        import fastcluster
        from scipy.cluster.hierarchy import fcluster
        from scipy.cluster.hierarchy import linkage

        x = x.view(x.shape[0], -1).transpose(-2, -1).to("cpu").detach()
        Z = fastcluster.average(x)
        label = fcluster(Z, t=n_clusters, criterion="maxclust")
        return label.reshape(h, w)
    elif method == "kmeans":
        kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init="auto").fit(
            x.view(x.shape[0], -1).transpose(-2, -1).to("cpu").detach()
        )
        label = kmeans.labels_
        return label.reshape(h, w)

    else:
        raise ValueError("Clustering method not found")


def eval(
    model,
    images,
    gt,
    method="agglomerative",
    n_clusters=7,
    saccade_r=1,
    pca=False,
    pca_dim=128,
):
    preds = []
    N = images.shape[0]
    _imgs, _ = gen_saccade_imgs(images, model.psize, model.psize // saccade_r)
    outputs = []
    for img in _imgs:
        v = model_preds(model, img)
        outputs.append(v.detach().cpu())

    nh, nw = int(np.sqrt(len(_imgs))), int(np.sqrt(len(_imgs)))
    ho, wo = outputs[0].shape[-2], outputs[0].shape[-1]
    nimg = torch.zeros(N, outputs[0].shape[1], ho, nh, wo, nw)
    for h in range(nh):
        for w in range(nw):
            nimg[:, :, :, h, :, w] = outputs[h * (nh) + w]
    nimg = nimg.view(N, -1, ho * nh, wo * nw)

    from source.utils import apply_pca_torch

    with torch.no_grad():

        if pca:
            if USE_GPU_FOR_PCA:
                nimg = nimg.to('cuda')
            pcaimg_ = apply_pca_torch(nimg, n_components=pca_dim)
            x = pcaimg_.to('cpu') if USE_GPU_FOR_PCA else pcaimg_

        else:
            x = nimg

    for idx in range(N):
        _x = x[idx]
        pred = clustering(_x, *_x.shape[1:], method, n_clusters)
        pred = torch.nn.Upsample(
            scale_factor=(images.shape[-2]/pred.shape[-2],
                          images.shape[-1]/pred.shape[-1]),
            mode='nearest')(torch.Tensor(pred[None, None]).float())[0, 0]
        preds.append(pred)

    preds = torch.stack(preds, 0).long()

    scores = {}
    evaluate_sem = False
    if isinstance(gt, list):
        gt_sem = gt[1]
        gt = gt[0]
        evaluate_sem = True

    _gt = ((gt > 0).float() * gt).long()  # set ignore bg (-1) to 0
    # compute fgari
    scores["fgari"] = np.array(calc_fgari_score(_gt, preds))

    # compute mean best overlap
    score, _scores = calc_mean_best_overlap(gt.numpy(), preds.numpy())
    scores["mbo"] = score
    scores["mbo_scores"] = _scores

    if evaluate_sem:
        score, _scores = calc_mean_best_overlap(gt_sem.numpy(), preds.numpy())
        scores["mbo_c"] = score
        scores["mbo_c_scores"] = _scores

    return scores, preds


def get_loader(data, data_root, imsize, batchsize):

    from source.data.datasets.objs.load_data import load_data

    dataset, imsize, collate_fn = load_data(
        data, data_root, imsize, is_eval=True)

    if data == "clevrtex_full" or data == "clevrtex_outd" or data == "clevrtex_camo":
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batchsize,
            num_workers=0,
            shuffle=True,
            collate_fn=collate_fn,
        )

    elif data == "coco":
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batchsize,
            num_workers=0,
            shuffle=True,
            collate_fn=collate_fn,
        )
    else:
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batchsize,
            num_workers=0,
            shuffle=True,
        )
    return loader, imsize


def eval_dataset(
    model,
    data,
    data_root=None,
    imsize=None,
    batchsize=100,
    method="agglomerative",
    instance=True,
    saccade_r=1,
    pca=False,
):

    scores = []
    preds = []
    masks = []

    loader, imsize = get_loader(data, data_root, imsize, batchsize)
    for ret in tqdm.tqdm(loader):
        pca_dim = 128
        if data == "clevr":
            images = ret[0]
            if instance:
                labels = ret[1]["pixelwise_instance_labels"]
            else:
                labels = ret[1]["pixelwise_class_labels"]
            n_clusters = 11
        elif data == 'tetrominoes':
            images = ret[0]
            labels = ret[1]["pixelwise_instance_labels"]
            n_clusters = 4
        elif data == 'dsprites':
            images = ret[0]
            labels = ret[1]["pixelwise_instance_labels"]
            n_clusters = 7
        elif data == "clevrtex_camo" or data == "clevrtex_full" or data == "clevrtex_outd":
            images = ret[1]
            labels = ret[2][:, 0]
            n_clusters = 11

        elif data == "pascal":
            images = ret[0]
            labels_instance = ret[1]["pixelwise_instance_labels"]
            labels_sem = ret[1]["pixelwise_class_labels"]
            labels = [labels_instance, labels_sem]
            n_clusters = 4

        elif data == "coco":
            images = ret["img"]
            labels_instance = ret["masks"].long()
            labels_sem = ret["sem_masks"].long()
            ovlp = ret["inst_overlap_masks"].long()
            labels_instance[ovlp == 1] = -1
            labels_sem[ovlp == 1] = -1
            labels = [labels_instance, labels_sem]
            n_clusters = 7
        score, pred = eval(
            model,
            images,
            labels,
            method,
            n_clusters,
            saccade_r=saccade_r,
            pca=pca,
            pca_dim=pca_dim,
        )
        scores.append(score)
        preds.append(pred)
        masks.append(labels)
    return scores, preds


def print_stats(scores):
    fgaris = []
    mbos = []
    mbocs = []
    for _s in scores:
        fgaris.append(_s["fgari"])
        mbos.append(_s["mbo_scores"])
        if "mbo_c" in _s:
            mbocs.append(_s["mbo_c_scores"])
    print("fgari: ", np.concatenate(fgaris, 0).mean(), np.concatenate(fgaris, 0).std())
    _mbos = np.concatenate(mbos)
    _mbos = _mbos[_mbos != -1]
    print("mbo: ", np.mean(_mbos), np.std(_mbos))
    if len(mbocs) > 0:
        _mbocs = np.concatenate(mbocs)
        _mbocs = _mbocs[_mbocs != -1]
        print("mbo_c: ", np.mean(_mbocs), np.std(_mbocs))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # Eval options
    parser.add_argument("--model_path", type=str, help="path to the model")
    parser.add_argument("--saccade_r", type=int,
                        default=1, help="Uptiling factor")
    parser.add_argument("--pca", type=str2bool, default=True)

    # Data loading
    parser.add_argument("--limit_cores_used", type=str2bool, default=False)
    parser.add_argument("--cpu_core_start", type=int,
                        default=0, help="start core")
    parser.add_argument("--cpu_core_end", type=int,
                        default=32, help="end core")
    parser.add_argument("--data", type=str, default="clevrtex_full")
    parser.add_argument(
        "--data_root",
        type=str,
        default=None,
        help="optional. you can specify the dir path if the default path of each dataset is not appropritate one. Currently only applied to ImageNet",
    )
    parser.add_argument("--batchsize", type=int, default=250)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument(
        "--data_imsize",
        type=int,
        default=None,
        help="Image size. If None, use the default size of each dataset",
    )

    # General model options
    parser.add_argument("--model", type=str, default="knet", help="model")
    parser.add_argument("--L", type=int, default=2, help="num of layers")
    parser.add_argument("--ch", type=int, default=256, help="num of channels")
    parser.add_argument(
        "--model_imsize",
        type=int,
        default=None,
        help="""
        Model's imsize that was set when it was initialized. 
        This is used when evaluating or when finetuning a pretrained model.
        """,
    )
    parser.add_argument("--autorescale", type=str2bool, default=False)
    parser.add_argument("--psize", type=int, default=8, help="patch size")
    parser.add_argument("--ksize", type=int, default=1, help="kernel size")
    parser.add_argument("--T", type=int, default=8, help="num of recurrence")
    parser.add_argument(
        "--maxpool", type=str2bool, default=True, help="max pooling or avg pooling"
    )
    parser.add_argument(
        "--heads", type=int, default=8, help="num of heads in self-attention"
    )
    parser.add_argument(
        "--gta",
        type=str2bool,
        default=True,
        help="""
        use Geometric Transform Attention (https://github.com/autonomousvision/gta) as positional encoding.
        If False, use standard absolute positional encoding
        """,
    )

    # AKOrN options
    parser.add_argument("--N", type=int, default=4,
                        help="num of rotating dimensions")
    parser.add_argument("--J", type=str, default="conv", help="connectivity")
    parser.add_argument("--use_omega", type=str2bool, default=False)
    parser.add_argument("--global_omg", type=str2bool, default=False)
    parser.add_argument(
        "--c_norm",
        type=str,
        default="gn",
        help="normalization. gn, sandb(scale and bias), or none",
    )

    parser.add_argument(
        "--use_ro_x",
        type=str2bool,
        default=False,
        help="apply linear transform to oscillators between consecutive layers",
    )

    # ablation of some components in the AKOrN's block
    parser.add_argument(
        "--no_ro", type=str2bool, default=False, help="ablation: no use readout module"
    )
    parser.add_argument(
        "--project",
        type=str2bool,
        default=True,
        help="use projection or not in the Kuramoto layer",
    )

    args = parser.parse_args()

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.enable_flash_sdp(enabled=True)

    if args.limit_cores_used:
        def worker_init_fn(worker_id):
            os.sched_setaffinity(
                0, range(args.cpu_core_start, args.cpu_core_end))

    if args.model == "akorn":
        net = AKOrN(
            args.N,
            ch=args.ch,
            L=args.L,
            T=args.T,
            J=args.J,  # "conv" or "attn",
            use_omega=args.use_omega,
            global_omg=args.global_omg,
            c_norm=args.c_norm,
            psize=args.psize,
            imsize=args.model_imsize,
            autorescale=args.autorescale,
            maxpool=args.maxpool,
            project=args.project,
            heads=args.heads,
            use_ro_x=args.use_ro_x,
            no_ro=args.no_ro,
            gta=args.gta,
        ).to("cuda")
    elif args.model == "vit":
        net = ViT(
            psize=args.psize,
            imsize=args.model_imsize,
            autorescale=args.autorescale,
            ch=args.ch,
            blocks=args.L,
            heads=args.heads,
            mlp_dim=2 * args.ch,
            T=args.T,
            maxpool=args.maxpool,
            gta=args.gta,
        ).cuda()

    model = EMA(net)
    model.load_state_dict(torch.load(
        args.model_path, weights_only=True)["model_state_dict"])
    model = model.ema_model

    with torch.no_grad():
        scores, preds = eval_dataset(
            model,
            data=args.data,
            data_root=args.data_root,
            imsize=args.data_imsize,
            batchsize=args.batchsize,
            instance=True,
            method="agglomerative",
            saccade_r=args.saccade_r,
            pca=args.pca,
        )
        print_stats(scores)
