import argparse
import logging
import os
import time
from collections import OrderedDict
from datetime import datetime

import numpy as np

import ml_collections
import torch
import torch.nn as nn
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP

import utils
from utils import Key
from modules import DeepONet, FullyConnected, ModifiedMlp, ResNet
from data.loader import create_loader
from data.dataset import MioDataset, preprocess

_logger = logging.getLogger("train")

# The first arg parser parses out only the --config argument, this argument is used to
# load a yaml file containing key-values that override the defaults
# for the main parser below
config_parser = parser = argparse.ArgumentParser(
    description="Training Config", add_help=False
)
parser.add_argument(
    "-c",
    "--config",
    default="",
    type=str,
    metavar="FILE",
    help="YAML config file specifying default arguments",
)


def _get_config():
    # Do we have a config file to parse?
    args_config, _ = config_parser.parse_known_args()
    if args_config.config:
        with open(args_config.config, "r") as f:
            cfg = yaml.safe_load(f)
            cfg = ml_collections.ConfigDict(cfg)

        # with open(cfg.model_config_path, "r") as f:
        #     model_cfg = yaml.safe_load(f)
        #     cfg.model = ml_collections.ConfigDict(model_cfg)
    else:
        raise ValueError("Must specify a config file")

    return cfg


def main():
    utils.setup_default_logging()
    cfg = _get_config()
    # print(f"Number of available CUDA devices: {torch.cuda.device_count()}")

    # TODO (hard): improve multiple processes for distributed training
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    device = utils.init_distributed_device(cfg)

    if cfg.distributed:
        _logger.info(
            "Training in distributed mode with multiple processes, 1 device per process."
            f"Process {cfg.rank}, total {cfg.world_size}, device {cfg.device}."
        )
    else:
        _logger.info(f"Training with a single process on 1 device ({cfg.device}).")
    assert cfg.rank >= 0

    utils.random_seed(cfg.seed, 0)

    # TODO: function to create the dataset
    assert cfg.data.train_data, "train_data must be specified"
    np_data = dict(np.load(cfg.data.train_data, allow_pickle=True))
    branch_dr, trunk_dr, label_dr, input_shape_dict = preprocess(cfg, np_data)

    latent_size = cfg.model.latent_size

    def _get_activation():
        act = cfg.model.get("activation", "relu")
        if act == "tanh":
            return nn.Tanh
        elif act == "relu":
            return nn.ReLU

    def create_model(model_cfg, shape_dict):
        input_name = model_cfg.get("input_key")
        model_type = model_cfg.get("type")
        if model_type == "mlp":
            net = FullyConnected(
                [Key(input_name, size=shape_dict[input_name])],
                [Key(model_cfg.get("output_key"), latent_size)],
                model_cfg.hidden_units,
                activation=_get_activation(),
            )
        elif model_type == "modified_mlp":
            net = ModifiedMlp(
                [Key(input_name, size=shape_dict[input_name])],
                [Key(model_cfg.get("output_key"), latent_size)],
                model_cfg.hidden_units,
                activation=_get_activation(),
            )
        elif model_type == "resnet":
            net = ResNet(
                [Key(input_name, size=shape_dict[input_name])],
                [Key(model_cfg.get("output_key"), latent_size)],
                model_cfg.hidden_units,
                activation=_get_activation(),
            )
        return net

    branch_net_list = []
    for k, d in cfg.model.items():
        if "branch" in k:
            branch_net_list.append(create_model(d, input_shape_dict["branch"]))
    trunk_net = create_model(cfg.model.trunk_net, input_shape_dict["trunk"])

    model = DeepONet(branch_net_list, trunk_net, output_keys=[Key("psi", 1)])

    # TODO (hard): check if this is correct
    if cfg.initial_checkpoint:
        model = utils.load_checkpoint(model, cfg.initial_checkpoint)

    if utils.is_primary(cfg):
        _logger.info(
            f"Model created, param count:{sum([m.numel() for m in model.parameters()])}"
        )

    model.to(device=device)

    # TODO: function to create the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.optimizer.lr)

    # setup synchronized BatchNorm for distributed training
    if cfg.distributed and cfg.sync_bn:
        cfg.dist_bn = ""
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        if utils.is_primary(cfg):
            _logger.info(
                "Converted model to use Synchronized BatchNorm. WARNING: You may have issues if using "
                "zero initialized BN layers (enabled by default for ResNets) while sync-bn enabled."
            )

    if cfg.distributed:
        if utils.is_primary(cfg):
            _logger.info("Using native Torch DistributedDataParallel.")
        model = DDP(model, device_ids=[device], find_unused_parameters=True)

    dataset_train = MioDataset(
        branch_dr,
        trunk_dr,
        label_dr,
        collocation_size=cfg.data.collocation_size,
    )

    loader_train = create_loader(
        dataset_train,
        batch_size=cfg.training.train_batch_size,
        is_training=True,
        num_workers=cfg.num_workers,
        distributed=cfg.distributed,
    )

    eval_data = dict(np.load(cfg.data.eval_data, allow_pickle=True))
    eval_branch_dr, eval_trunk_dr, eval_label_dr, _ = preprocess(cfg, eval_data)

    dataset_eval = MioDataset(
        eval_branch_dr,
        eval_trunk_dr,
        eval_label_dr,
        collocation_size=cfg.data.collocation_size,
    )

    loader_eval = create_loader(
        dataset_eval,
        batch_size=cfg.training.eval_batch_size,
        is_training=True,
        num_workers=cfg.num_workers,
        distributed=cfg.distributed,
    )

    # TODO (hard): torchscript to speed up training

    # TODO: set up loss function
    train_loss_fn = nn.MSELoss()
    train_loss_fn = train_loss_fn.to(device=device)
    validate_loss_fn = nn.MSELoss().to(device=device)

    saver = None
    output_dir = None
    best_metric = None

    if utils.is_primary(cfg):
        if cfg.experiment:
            exp_name = cfg.experiment
        else:
            exp_name = "-".join(
                [
                    datetime.now().strftime("%Y%m%d-%H%M%S"),
                    "deeponet",
                ]
            )
        output_dir = utils.get_outdir(
            cfg.output if cfg.output else "./output/train", exp_name
        )

        saver = utils.CheckpointSaver(
            model=model,
            optimizer=optimizer,
            config=cfg,
            checkpoint_dir=output_dir,
            recovery_dir=output_dir,
            decreasing=True,
            max_history=cfg.checkpoint_hist,
        )
        with open(os.path.join(output_dir, "config.yaml"), "w") as f:
            f.write(yaml.dump(cfg.to_dict()))

    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer=optimizer,
        step_size=cfg.optimizer.step_size,
        gamma=cfg.optimizer.gamma,
    )

    start_epoch = 0
    if cfg.training.start_epoch is not None:
        start_epoch = cfg.training.start_epoch

    for epoch in range(start_epoch, cfg.training.epochs):
        # print(epoch)
        if hasattr(dataset_train, "set_epoch"):
            dataset_train.set_epoch(epoch)
        elif cfg.distributed and hasattr(loader_train.sampler, "set_epoch"):
            loader_train.sampler.set_epoch(epoch)

        train_metrics = train_one_epoch(
            epoch,
            model,
            loader_train,
            optimizer,
            train_loss_fn,
            cfg,
            lr_scheduler=lr_scheduler,
            saver=saver,
            output_dir=output_dir,
        )

        eval_metrics = validate(
            model,
            loader_eval,
            validate_loss_fn,
            cfg,
        )

        if output_dir is not None:
            lrs = [param_group["lr"] for param_group in optimizer.param_groups]
            utils.update_summary(
                epoch,
                train_metrics,
                eval_metrics,
                filename=os.path.join(output_dir, "summary.csv"),
                lr=sum(lrs) / len(lrs),
                write_header=best_metric is None,
            )

        if saver is not None:
            # save proper checkpoint with eval metric
            save_metric = eval_metrics["loss"]
            best_metric, _ = saver.save_checkpoint(epoch, metric=save_metric)


def train_one_epoch(
    epoch,
    model,
    loader,
    optimizer,
    loss_fn,
    args,
    device=torch.device("cuda"),
    lr_scheduler=None,
    saver=None,
    output_dir=None,
):
    second_order = hasattr(optimizer, "is_second_order") and optimizer.is_second_order
    update_time_m = utils.AverageMeter()
    data_time_m = utils.AverageMeter()
    losses_m = utils.AverageMeter()
    rmse_m = utils.AverageMeter()

    # print(f"Epoch {epoch} started")

    model.train()
    data_start_time = update_start_time = time.time()
    optimizer.zero_grad()
    update_sample_count = 0

    try:
        num_batches = len(loader)
    except StopIteration:
        num_batches = 1000

    iterator = iter(loader)

    def rmse_loss(pre, label):
        return torch.sqrt(torch.sum((pre - label) ** 2) / torch.sum(label**2))

    # for batch_idx, input in enumerate(loader):
    for batch_idx in range(num_batches):
        input = next(iterator)
        input = {k: v.to(device) for k, v in input.items()}
        data_time_m.update(time.time() - data_start_time)

        def _forward():
            output = model(input)
            loss = loss_fn(output["psi"], input["psi_label"])
            rmse = rmse_loss(output["psi"], input["psi_label"])
            return loss, rmse

        def _backward(_loss):
            _loss.backward(create_graph=second_order)
            optimizer.step()

        loss, rmse = _forward()
        _backward(loss)

        if not args.distributed:
            losses_m.update(loss.item(), input["psi_label"].size(0))
            rmse_m.update(rmse.item(), input["psi_label"].size(0))

        update_sample_count += input["psi_label"].size(0)

        optimizer.zero_grad()

        time_now = time.time()
        update_time_m.update(time.time() - update_start_time)
        update_start_time = time_now
        # print(epoch % args.training.log_interval)
        if batch_idx % args.training.log_interval == 0:
            if args.distributed:
                reduced_loss = utils.reduce_tensor(loss.data, args.world_size)
                reduced_rmse = utils.reduce_tensor(rmse, args.world_size)
                losses_m.update(reduced_loss.item(), input["psi_label"].size(0))
                rmse_m.update(reduced_rmse.item(), input["psi_label"].size(0))
                update_sample_count *= args.world_size

            # print(utils.is_primary(args))

            if utils.is_primary(args):
                _logger.info(
                    f"Train: {epoch}[{batch_idx:>4d}/{num_batches-1}] "
                    f"Loss: {losses_m.val:#.3g} ({losses_m.avg:#.3g})  "
                    f"RMSE: {rmse_m.val:#.3g} ({rmse_m.avg:#.3g})  "
                    f"Time: {update_time_m.val:.3f}s, {update_sample_count / update_time_m.val:>7.2f}/s  "
                    f"({update_time_m.avg:.3f}s, {update_sample_count / update_time_m.avg:>7.2f}/s)  "
                    f"Data: {data_time_m.val:.3f} ({data_time_m.avg:.3f})"
                )

        if saver is not None and args.training.recovery_interval:
            saver.save_recovery(epoch, batch_idx=batch_idx)

        if lr_scheduler is not None:
            lr_scheduler.step()

        update_sample_count = 0
        data_start_time = time.time()
        # end for

    if hasattr(optimizer, "sync_lookahead"):
        optimizer.sync_lookahead()

    return OrderedDict([("loss", losses_m.avg), ("RMSE", rmse_m.avg)])


def validate(
    model,
    loader,
    loss_fn,
    args,
    device=torch.device("cuda"),
    log_suffix="",
):
    losses_m = utils.AverageMeter()
    rmse_m = utils.AverageMeter()

    num_batches = len(loader)
    model.eval()

    iterator = iter(loader)

    def rmse_loss(pre, label):
        return torch.sqrt(torch.sum((pre - label) ** 2) / torch.sum(label**2))

    with torch.no_grad():
        # for batch_idx, input in enumerate(val_loader):
        for batch_idx in range(num_batches):
            last_batch = batch_idx == (num_batches - 1)
            input = next(iterator)
            input = {k: v.to(device) for k, v in input.items()}
            output = model(input)
            loss = loss_fn(output["psi"], input["psi_label"])
            rmse = rmse_loss(output["psi"], input["psi_label"])

            losses_m.update(loss.item(), input["psi_label"].size(0))
            rmse_m.update(rmse.item(), input["psi_label"].size(0))

            if utils.is_primary(args) and (
                last_batch or batch_idx % args.training.log_interval == 0
            ):
                log_name = f"Test{log_suffix}"
                _logger.info(
                    f"{log_name}: [{batch_idx:>4d}/{num_batches-1}]  "
                    f"Loss: {losses_m.val:#.3g} ({losses_m.avg:#.3g})  "
                    f"RMSE: {rmse_m.val:#.3g} ({rmse_m.avg:#.3g})  "
                )

    metrics = OrderedDict([("loss", losses_m.avg), ("RMSE", rmse_m.avg)])
    return metrics


if __name__ == "__main__":
    main()
