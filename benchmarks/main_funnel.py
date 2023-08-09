import sys, os
from typing import Callable
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from functools import partial
import argparse
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_pruning as tp
import registry
import engine.utils as utils

from ptflops import get_model_complexity_info

import numpy as np
import copy
from collections import OrderedDict

parser = argparse.ArgumentParser()

# Basic options
parser.add_argument("--mode", type=str, required=True, choices=["pretrain", "prune", "test"])
parser.add_argument("--model", type=str, required=True)
parser.add_argument("--verbose", action="store_true", default=False)
parser.add_argument("--dataset", type=str, default="cifar100", choices=['cifar10', 'cifar100', 'modelnet40'])
parser.add_argument("--batch-size", type=int, default=128)
parser.add_argument("--total-epochs", type=int, default=100)
parser.add_argument("--lr-decay-milestones", default="60,80", type=str, help="milestones for learning rate decay")
parser.add_argument("--lr-decay-gamma", default=0.1, type=float)
parser.add_argument("--lr", default=0.01, type=float, help="learning rate")
parser.add_argument("--restore", type=str, default=None)
parser.add_argument('--output-dir', default='run', help='path where to save')

# For pruning
parser.add_argument("--method", type=str, default=None)
parser.add_argument("--speed-up", type=float, default=2)
parser.add_argument("--global-speed-up", action="store_true", default=False)
parser.add_argument("--max-accuracy-drop", type=float, default=None)
parser.add_argument("--acc-drop-iters", type=int, default=1)
parser.add_argument("--max-sparsity", type=float, default=1.0)
parser.add_argument("--soft-keeping-ratio", type=float, default=0.0)
parser.add_argument("--reg", type=float, default=5e-4)
parser.add_argument("--weight-decay", type=float, default=5e-4)

parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--global-pruning", action="store_true", default=False)
parser.add_argument("--sl-total-epochs", type=int, default=100, help="epochs for sparsity learning")
parser.add_argument("--sl-lr", default=0.01, type=float, help="learning rate for sparsity learning")
parser.add_argument("--sl-lr-decay-milestones", default="60,80", type=str, help="milestones for sparsity learning")
parser.add_argument("--sl-reg-warmup", type=int, default=0, help="epochs for sparsity learning")
#parser.add_argument("--sl_restore", type=str, default=None)
parser.add_argument("--sl-restore", action="store_true", default=False)
parser.add_argument("--iterative-steps", default=400, type=int)

args = parser.parse_args()

def progressive_pruning(pruner, model, example_inputs, speed_up, global_speed_up=False, tail_modules=[]):
    model.eval()
    if not global_speed_up:
        assert tail_modules != []
        base_ops, _ = tp.utils.count_ops_and_params(model, example_inputs=example_inputs, ignore_modules=tail_modules)
    else:
        base_ops, _ = tp.utils.count_ops_and_params(model, example_inputs=example_inputs)
    current_speed_up = 1
    while current_speed_up < speed_up:
        pruner.step()
        if not global_speed_up:
            pruned_ops, _ = tp.utils.count_ops_and_params(model, example_inputs=example_inputs, ignore_modules=tail_modules)
        else:
            pruned_ops, _ = tp.utils.count_ops_and_params(model, example_inputs=example_inputs)
        current_speed_up = float(base_ops) / pruned_ops
    return current_speed_up

def progressive_pruning_2(pruner, model, test_loader, device, min_tolerable_accuracy):
    model.eval()
    while eval(model, test_loader, device=device)[0] > min_tolerable_accuracy:
        pruner.step()

def eval(model, test_loader, device=None):
    correct = 0
    total = 0
    loss = 0
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    with torch.no_grad():
        for i, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            out = model(data)
            loss += F.cross_entropy(out, target, reduction="sum")
            pred = out.max(1)[1]
            correct += (pred == target).sum()
            total += len(target)
    return (correct / total).item(), (loss / total).item()

def train_model(
    model,
    train_loader,
    test_loader,
    epochs,
    lr,
    lr_decay_milestones,
    lr_decay_gamma=0.1,
    save_as=None,
    
    # For pruning
    weight_decay=5e-4,
    save_state_dict_only=True,
    regularizer=None,
    device=None,

    # Training will terminate if the target accuracy is achieved
    target_accuracy=None,
    last_epoch=-1
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=0.9,
        weight_decay=weight_decay if regularizer is None else 0.0,
    )
    if lr_decay_milestones != "":
        milestones = [int(ms) for ms in lr_decay_milestones.split(",")]
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=milestones, gamma=lr_decay_gamma
        )
    else:
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=lr_decay_gamma
        )
    for i in range(last_epoch + 1):
        scheduler.step()
    model.to(device)
    best_acc = -1
    for epoch in range(epochs):
        model.train()
        for i, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            out = model(data)
            loss = F.cross_entropy(out, target)
            loss.backward()
            if regularizer is not None:
                regularizer(model) # for sparsity learning
            optimizer.step()
            if i % 10 == 0 and args.verbose:
                args.logger.info(
                    "Epoch {:d}/{:d}, iter {:d}/{:d}, loss={:.4f}, lr={:.4f}".format(
                        epoch,
                        epochs,
                        i,
                        len(train_loader),
                        loss.item(),
                        optimizer.param_groups[0]["lr"],
                    )
                )
        
        model.eval()
        acc, val_loss = eval(model, test_loader, device=device)
        args.logger.info(
            "Epoch {:d}/{:d}, Acc={:.4f}, Val Loss={:.4f}, lr={:.4f}".format(
                epoch, epochs, acc, val_loss, optimizer.param_groups[0]["lr"]
            )
        )
        if best_acc < acc:
            os.makedirs(args.output_dir, exist_ok=True)
            if args.mode == "prune":
                if save_as is None:
                    save_as = os.path.join( args.output_dir, "{}_{}_{}.pth".format(args.dataset, args.model, args.method) )

                if save_state_dict_only:
                    torch.save(model.state_dict(), save_as)
                else:
                    torch.save(model, save_as)
            elif args.mode == "pretrain":
                if save_as is None:
                    save_as = os.path.join( args.output_dir, "{}_{}.pth".format(args.dataset, args.model) )
                torch.save(model.state_dict(), save_as)
            best_acc = acc
        scheduler.step()

        if target_accuracy != None:
            if acc > target_accuracy:
                return epoch + 1
        
    args.logger.info("Best Acc=%.4f" % (best_acc))
    return epoch + 1


def get_pruner(model, example_inputs, ch_sparsity_dict={}, customized_pruners=None):
    sparsity_learning = False
    if args.method == "random":
        imp = tp.importance.RandomImportance()
        pruner_entry = partial(tp.pruner.MagnitudePruner, global_pruning=args.global_pruning)
    elif args.method == "l1":
        imp = tp.importance.MagnitudeImportance(p=1)
        pruner_entry = partial(tp.pruner.MagnitudePruner, global_pruning=args.global_pruning)
    elif args.method == "lamp":
        imp = tp.importance.LAMPImportance(p=2)
        pruner_entry = partial(tp.pruner.MagnitudePruner, global_pruning=args.global_pruning)
    elif args.method == "slim":
        sparsity_learning = True
        imp = tp.importance.BNScaleImportance()
        pruner_entry = partial(tp.pruner.BNScalePruner, reg=args.reg, global_pruning=args.global_pruning)
    elif args.method == "group_norm":
        imp = tp.importance.GroupNormImportance(p=2)
        pruner_entry = partial(tp.pruner.GroupNormPruner, global_pruning=args.global_pruning)
    elif args.method == "group_sl":
        sparsity_learning = True
        imp = tp.importance.GroupNormImportance(p=2)
        pruner_entry = partial(tp.pruner.GroupNormPruner, reg=args.reg, global_pruning=args.global_pruning)
    else:
        raise NotImplementedError
    
    #args.is_accum_importance = is_accum_importance
    unwrapped_parameters = []
    args.sparsity_learning = sparsity_learning
    ignored_layers = []

    # ignore output layers
    for m in model.modules():
        if isinstance(m, torch.nn.Linear) and m.out_features == args.num_classes:
            ignored_layers.append(m)
        elif isinstance(m, torch.nn.modules.conv._ConvNd) and m.out_channels == args.num_classes:
            ignored_layers.append(m)
    
    # Here we fix iterative_steps=200 to prune the model progressively with small steps 
    # until the required speed up is achieved.
    pruner = pruner_entry(
        model,
        example_inputs,
        importance=imp,
        iterative_steps=args.iterative_steps,
        ch_sparsity=1.0,
        ch_sparsity_dict=ch_sparsity_dict,
        max_ch_sparsity=args.max_sparsity,
        ignored_layers=ignored_layers,
        customized_pruners=customized_pruners,
        unwrapped_parameters=unwrapped_parameters,
    )
    return pruner


def main():
    if args.seed is not None:
        torch.manual_seed(args.seed)

    # Logger
    if args.mode == "prune":
        prefix = 'global' if args.global_pruning else 'local'
        logger_name = "{}-{}-{}-{}".format(args.dataset, prefix, args.method, args.model)
        args.output_dir = os.path.join(args.output_dir, args.dataset, args.mode, logger_name)
        log_file = "{}/{}.txt".format(args.output_dir, logger_name)
    elif args.mode == "pretrain":
        args.output_dir = os.path.join(args.output_dir, args.dataset, args.mode)
        logger_name = "{}-{}".format(args.dataset, args.model)
        log_file = "{}/{}.txt".format(args.output_dir, logger_name)
    elif args.mode == "test":
        log_file = None
    args.logger = utils.get_logger(logger_name, output=log_file)

    # Model & Dataset
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes, train_dst, val_dst, input_size = registry.get_dataset(
        args.dataset, data_root="data"
    )
    args.num_classes = num_classes
    model = registry.get_model(args.model, num_classes=num_classes, pretrained=True, target_dataset=args.dataset)
    train_loader = torch.utils.data.DataLoader(
        train_dst,
        batch_size=args.batch_size,
        num_workers=4,
        drop_last=True,
        shuffle=True,
    )
    test_loader = torch.utils.data.DataLoader(
        val_dst, batch_size=args.batch_size, num_workers=4
    )
    
    for k, v in utils.utils.flatten_dict(vars(args)).items():  # print args
        args.logger.info("%s: %s" % (k, v))

    if args.restore is not None:
        loaded = torch.load(args.restore, map_location="cpu")
        if isinstance(loaded, nn.Module):
            model = loaded
        else:
            model.load_state_dict(loaded)
        args.logger.info("Loading model from {restore}".format(restore=args.restore))
    model = model.to(args.device)

    # @!
    import copy
    ori_model = copy.deepcopy(model)

    ######################################################
    # Training / Pruning / Testing
    example_inputs = train_dst[0][0].unsqueeze(0).to(args.device)
    if args.mode == "pretrain":
        ops, params = tp.utils.count_ops_and_params(
            model, example_inputs=example_inputs,
        )
        args.logger.info("Params: {:.2f} M".format(params / 1e6))
        args.logger.info("ops: {:.2f} M".format(ops / 1e6))
        train_model(
            model=model,
            epochs=args.total_epochs,
            lr=args.lr,
            lr_decay_milestones=args.lr_decay_milestones,
            train_loader=train_loader,
            test_loader=test_loader
        )
    elif args.mode == "prune":
        model.eval()
        ori_ops, ori_size = tp.utils.count_ops_and_params(model, example_inputs=example_inputs)
        ori_acc, ori_val_loss = eval(model, test_loader, device=args.device)

        ch_sparsity_dict = OrderedDict()
        conv_idx = 0.0

        first_tail_module = 'unknown'
        if 'vgg' in args.model:
            first_tail_module = 'block2.1'
        elif 'resnet' in args.model:
            first_tail_module = 'layer3'

        skip_modules = []
        for name, module in model.named_modules():
            if first_tail_module in name:
                break
            if module in skip_modules:
                continue
            if isinstance(module, torch.nn.Conv2d):
                conv_idx += 1
                ch_sparsity_dict[module] = conv_idx
            elif 'shortcut' in name:
                ch_sparsity_dict[module] = conv_idx
                for cmodule in module.children():
                    skip_modules.append(cmodule)
        for module in ch_sparsity_dict.keys():
            ch_sparsity_dict[module] = ch_sparsity_dict[module] / conv_idx

        # args.max_sparsity = 1 - (1 / 128)

        # for layer_idx in range(2):
        #     for block_idx in range(2):
        #         for conv_idx in range(2):
        #             ch_sparsity_dict[model.get_submodule(f'layer{layer_idx + 1}.{block_idx}.conv{conv_idx + 1}')] = layer_idx * 4 + block_idx * 2 + conv_idx + 1

        # for module in ch_sparsity_dict.keys():
        #     ch_sparsity_dict[module] /= len(ch_sparsity_dict)

        # for layer_idx in range(2):
        #     for block_idx in range(2):
        #         ch_sparsity_dict[model.get_submodule(f'layer{layer_idx + 1}.{block_idx}.shortcut')] = ch_sparsity_dict[model.get_submodule(f'layer{layer_idx + 1}.{block_idx}.conv2')]

        # ch_sparsity_dict = {}
        # conv_idx = 0
        # for m in model.modules():
        #     if isinstance(m, torch.nn.Conv2d):
        #         if conv_idx > 4:
        #             break
        #         conv_idx = conv_idx + 1
        #         ch_sparsity_dict[m] = conv_idx
        #     elif isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.ReLU):
        #         ch_sparsity_dict[m] = conv_idx
        # for m in ch_sparsity_dict.keys():
        #     ch_sparsity_dict[m] = ch_sparsity_dict[m] / float(conv_idx)

        # ch_sparsity_dict = {}
        # resblock_idx = 0
        # for child in model.children():
        #     for grandchild in child.children():
        #         resblock_idx = resblock_idx + 1
        #         ch_sparsity_dict[grandchild] = resblock_idx
        # for m in ch_sparsity_dict.keys():
        #     ch_sparsity_dict[m] = ch_sparsity_dict[m] / float(resblock_idx)

        customized_pruners = None
        for module in model.modules():
            if 'DropLayer'in str(module):
                drop_layer_pruner = registry.models.cifar.resnet_drop.DropLayerPruner()
                customized_pruners = {registry.models.cifar.resnet_drop.DropLayer: drop_layer_pruner}
                break

        ignore_modules = []
        for m in model.modules():
            if m not in ch_sparsity_dict.keys():
                ignore_modules.append(m)
    
        ori_ops_head, ori_size_head = get_model_complexity_info(model, input_res=input_size[1:], print_per_layer_stat=True, verbose=True, as_strings=False, ignore_modules=ignore_modules)
        ori_ops_head, ori_size_head = tp.utils.count_ops_and_params(model, example_inputs=example_inputs, ignore_modules=ignore_modules)

        pruner = get_pruner(model, example_inputs=example_inputs, ch_sparsity_dict=ch_sparsity_dict, customized_pruners=customized_pruners)

        # 0. Sparsity Learning
        if args.sparsity_learning:
            reg_pth = "reg_{}_{}_{}_{}.pth".format(args.dataset, args.model, args.method, args.reg)
            reg_pth = os.path.join( os.path.join(args.output_dir, reg_pth) )
            if not args.sl_restore:
                args.logger.info("Regularizing...")
                train_model(
                    model,
                    train_loader=train_loader,
                    test_loader=test_loader,
                    epochs=args.sl_total_epochs,
                    lr=args.sl_lr,
                    lr_decay_milestones=args.sl_lr_decay_milestones,
                    lr_decay_gamma=args.lr_decay_gamma,
                    regularizer=pruner.regularize,
                    save_state_dict_only=True,
                    save_as = reg_pth,
                )
            args.logger.info("Loading sparsity model from {}...".format(reg_pth))
            model.load_state_dict( torch.load( reg_pth, map_location=args.device) )
        
        # 1. Pruning
        model.eval()
        args.logger.info("Pruning...")
        if args.max_accuracy_drop != None:
            last_model = None
            acc_drop_steps = np.geomspace(0.0001, args.max_accuracy_drop, num=args.acc_drop_iters)[1:]
            no_more_space = False
            for acc_drop_step in acc_drop_steps:
                last_epoch = -1
                while True:
                    del last_model
                    last_model = copy.deepcopy(model)
                    for m in reversed(ch_sparsity_dict.keys()):
                        if isinstance(m, nn.Conv2d):
                            if m.weight.shape[0] == 1:
                                print("No more space for pruning")
                                no_more_space = True
                            break
                    if no_more_space:
                        break
                    progressive_pruning_2(pruner, model, test_loader, device=args.device, min_tolerable_accuracy=ori_acc - acc_drop_step)
                    # funnel_pth = "funnel_{}_{}_{}_{}.pth".format(args.dataset, args.model, args.method, args.max_accuracy_drop)
                    # funnel_pth = os.path.join( os.path.join(args.output_dir, funnel_pth) )
                    for m in ch_sparsity_dict.keys():
                        if isinstance(m, torch.nn.Conv2d):
                            for name, module in model.named_modules():
                                if module == m:
                                    ori_module = ori_model.get_submodule(name)
                                    print(f"{name}: {module.weight.shape[0]} / {ori_module.weight.shape[0]} = {module.weight.shape[0] / ori_module.weight.shape[0]}")
                                    break
                    finetune_epochs = train_model(
                        model,
                        epochs=args.total_epochs,
                        lr=args.lr,
                        lr_decay_milestones="",
                        lr_decay_gamma=0.95,
                        train_loader=train_loader,
                        test_loader=test_loader,
                        device=args.device,
                        save_state_dict_only=True,
                        # save_as=funnel_pth,
                        target_accuracy=ori_acc - acc_drop_step,
                        # last_epoch=last_epoch
                    )
                    last_epoch += finetune_epochs
                    if finetune_epochs == args.total_epochs:
                        # finetuning was not able to recover the original accuracy
                        # so switch back to the last model before last pruning step
                        print("Cannot recover accuracy")
                        break
            model = copy.deepcopy(last_model)
        else:
            progressive_pruning(pruner, model, example_inputs=example_inputs, speed_up=args.speed_up, global_speed_up=args.global_speed_up, tail_modules=ignore_modules)
        del pruner # remove reference
        args.logger.info(model)
        pruned_ops, pruned_size = tp.utils.count_ops_and_params(model, example_inputs=example_inputs)

        pruned_ops_head, pruned_size_head = get_model_complexity_info(model, input_res=input_size[1:], print_per_layer_stat=True, verbose=True, as_strings=False, ignore_modules=ignore_modules)
        pruned_ops_head, pruned_size_head = tp.utils.count_ops_and_params(model, example_inputs=example_inputs, ignore_modules=ignore_modules)

        pruned_acc, pruned_val_loss = eval(model, test_loader, device=args.device)
        
        args.logger.info(
            "Params: {:.2f} M => {:.2f} M ({:.2f}%)".format(
                ori_size / 1e6, pruned_size / 1e6, pruned_size / ori_size * 100
            )
        )
        args.logger.info(
            "FLOPs: {:.2f} M => {:.2f} M ({:.2f}%, {:.2f}X )".format(
                ori_ops / 1e6,
                pruned_ops / 1e6,
                pruned_ops / ori_ops * 100,
                ori_ops / pruned_ops,
            )
        )

        args.logger.info(
            "Head Params: {:.2f} M => {:.2f} M ({:.2f}%)".format(
                ori_size_head / 1e6, pruned_size_head / 1e6, pruned_size_head / ori_size_head * 100
            )
        )
        args.logger.info(
            "Head FLOPs: {:.2f} M => {:.2f} M ({:.2f}%, {:.2f}X )".format(
                ori_ops_head / 1e6,
                pruned_ops_head / 1e6,
                pruned_ops_head / ori_ops_head * 100,
                ori_ops_head / pruned_ops_head,
            )
        )

        args.logger.info("Acc: {:.4f} => {:.4f}".format(ori_acc, pruned_acc))
        args.logger.info(
            "Val Loss: {:.4f} => {:.4f}".format(ori_val_loss, pruned_val_loss)
        )
        
        # 2. Finetuning
        args.logger.info("Finetuning...")
        train_model(
            model,
            epochs=args.total_epochs,
            lr=args.lr,
            lr_decay_milestones=args.lr_decay_milestones,
            train_loader=train_loader,
            test_loader=test_loader,
            device=args.device,
            save_state_dict_only=False,
        )
    elif args.mode == "test":
        model.eval()
        ops, params = tp.utils.count_ops_and_params(
            model, example_inputs=example_inputs,
        )
        args.logger.info("Params: {:.2f} M".format(params / 1e6))
        args.logger.info("ops: {:.2f} M".format(ops / 1e6))
        acc, val_loss = eval(model, test_loader)
        args.logger.info("Acc: {:.4f} Val Loss: {:.4f}\n".format(acc, val_loss))

if __name__ == "__main__":
    main()
