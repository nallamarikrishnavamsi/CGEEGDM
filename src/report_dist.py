import torch
import numpy as np
from tqdm import tqdm
from model.classifier_pl import PLClassifier as PLClassifier_v2
from dataloader.TUEVDataset import TUEVDataset
from pyhealth.metrics.multiclass import multiclass_metrics_fn
from pyhealth.metrics.binary import binary_metrics_fn
from hydra.utils import instantiate
from sklearn import metrics
import gc
from functools import partial

from multiprocessing import Pool, Process, Manager
from pprint import pprint

def _eval(checkpoint, subset, config, device, return_dict):
    device = f"cuda:{device}"
    pl_cls=[None, None, PLClassifier_v2][config.get("pl_cls_version", 1)]
    
    is_binary = config.get("is_binary", False)
    if is_binary:
        assert config["n_class"] == 1
        metric_fn = binary_metrics_fn
        logit_to_prob_fn = lambda t: torch.nn.functional.sigmoid(t).unsqueeze(-1)
        prob_to_cls_fn = lambda a: (a >= 0.5).astype(int)
    else:
        metric_fn = multiclass_metrics_fn
        logit_to_prob_fn = partial(torch.nn.functional.softmax, dim=-1)
        prob_to_cls_fn = partial(np.argmax, axis=1)
    
    dataloader = torch.utils.data.DataLoader(
        subset,
        batch_size=config["batch_size"],
        num_workers=1,
        shuffle=False
    )

    data_count = len(subset)

    is_binary = config.get("is_binary", False)
    custom_metric = config.get("custom_metric", None)

    if custom_metric is not None: assert is_binary

    model = pl_cls.load_from_checkpoint(checkpoint, map_location=device)
    y_true = np.zeros((data_count * config.get("n_label_per_sample", 1)))
    y_prob = np.zeros((data_count * config.get("n_label_per_sample", 1), config["n_class"]))

    _idx = 0
    with torch.no_grad():
        for batch_input in tqdm(dataloader, total=data_count // config["batch_size"] + 1):
            batch_input = model.transfer_batch_to_device(batch_input, device, 0)

            _, pred, _ = model.get_loss_pred_label(batch_input, use_ema=True, data_is_cached=False)

            _bs = pred.shape[0]
            y_true[_idx: _idx + _bs] = batch_input[1].flatten().cpu().numpy()
            y_prob[_idx: _idx + _bs, :] = logit_to_prob_fn(pred).cpu().numpy()
            _idx += _bs

    if is_binary: y_prob = y_prob.flatten()
    
    result = metric_fn(y_true, y_prob, metrics=config["metrics"])
    
    result["conf"] = metrics.confusion_matrix(y_true, prob_to_cls_fn(y_prob))
    
    if custom_metric is not None:
        tn, fp, fn, tp = result["conf"].ravel()
        for name, form in custom_metric:
            result[name] = eval(form) # ...
    
    # result["checkpoint"] = checkpoint
    return_dict[checkpoint] = result
    
    del model
    gc.collect()
    torch.cuda.empty_cache()

    return result

def entry(config):
    checkpoint = config["checkpoint"]
    if isinstance(checkpoint, str): checkpoint = [checkpoint]
    
    is_binary = config.get("is_binary", False)
    custom_metric = config.get("custom_metric", None)

    if custom_metric is not None: assert is_binary
    
    dataset = TUEVDataset(
        config["data_dir"],
        schema=instantiate(config["schema"])
    )

    # with Pool(len(checkpoint)) as p:
        # all_result = p.map(_eval, [(c, s, config, i) for i, (c, s) in enumerate(zip(checkpoint, subsets))])
    manager = Manager()
    return_dict = manager.dict()
    ii = 0
    all_checkpoint = checkpoint
    print(len(all_checkpoint))
    while ii < len(all_checkpoint):
        checkpoint = all_checkpoint[ii: ii + 8]
        ii += 8
        processes = []

        for arg in [(c, dataset, config, i, return_dict) for i, c in enumerate(checkpoint)]:
            p = Process(target=_eval, args=arg)
            p.start()
            processes.append(p)

        for p in processes:
            p.join()
    
    all_result = return_dict.values()
    print(">>>>>>>>")
    # print(conf)
    for m in config["metrics"]:
        arr = np.array(list(map(lambda r: r[m], all_result))) * 100
        print(m, round(arr.mean().item(), 2), round(arr.std().item(), 2))
    
    if custom_metric is not None:
        for m, _ in custom_metric:
            arr = np.array(list(map(lambda r: r[m], all_result))) * 100
            print(m, round(arr.mean().item(), 2), round(arr.std().item(), 2))
    
    return_dict = dict(return_dict)
    pprint(return_dict)

    # FIXME assume the path does not contain "v"
    for ckpt_name, metrics in return_dict.items():
        v_idx = ckpt_name.find("v")
        if v_idx == -1: seed = "0"
        else: seed = ckpt_name[v_idx + 1:].split(".")[0]

        bacc = np.round(metrics["balanced_accuracy"] * 100, 2)
        kappa = np.round(metrics["cohen_kappa"] * 100, 2)
        wf1 = np.round(metrics["f1_weighted"] * 100, 2)

        print("", seed, kappa, bacc, f"{wf1} \\\\", sep=" & ")
    
    for ckpt_name, metrics in return_dict.items():
        v_idx = ckpt_name.find("v")
        if v_idx == -1: seed = "0"
        else: seed = ckpt_name[v_idx + 1:].split(".")[0]

        bacc = metrics["balanced_accuracy"] * 100
        kappa = metrics["cohen_kappa"] * 100
        wf1 = metrics["f1_weighted"] * 100

        print("", seed, kappa, bacc, f"{wf1} \\\\", sep=" & ")