import datetime
import logging
import os
import time
from typing import Any, Union, cast

import hydra
import lightning as L
import MinkowskiEngine as ME
import torch
import yaml
from hydra.utils import get_original_cwd, to_absolute_path
from lightning.fabric import Fabric
from lightning.fabric.loggers.csv_logs import CSVLogger
from lightning.fabric.loggers.tensorboard import TensorBoardLogger
from lightning.fabric.strategies.ddp import DDPStrategy
from omegaconf import DictConfig, OmegaConf
from torch import Tensor, nn
from torchmetrics.aggregation import RunningMean
from transformers import get_cosine_schedule_with_warmup

from emnet.config.registry import Config, register_configs
from emnet.geant.dataset import (
    electron_collate_fn,
    make_test_train_datasets,
    worker_init_fn,
)
from emnet.geant.io import convert_electron_pixel_file_to_hdf5
from emnet.networks.loss.criterion import EMCriterion
from emnet.networks.model import EMModel
from emnet.preprocessing import NSigmaSparsifyTransform
from emnet.utils.misc_utils import is_str_dict

_logger = logging.getLogger(__name__)

torch._dynamo.config.capture_scalar_outputs = True
# torch.set_float32_matmul_precision("high")


# register hydra configurations
register_configs()


@hydra.main(version_base=None, config_path="./conf", config_name="config")
def main(cfg: DictConfig):
    """Main run script."""
    # handle hydra config dict object
    if not isinstance(cfg, Config):
        config = OmegaConf.to_object(cfg)
        assert isinstance(config, Config)
    else:
        config = cfg

    # Set up logging
    _logger.setLevel(config.training.log_level)
    _logger.info("Starting...")
    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    if config.training.log_tensorboard:
        tb_logger = TensorBoardLogger(
            output_dir,
            name=(
                config.training.tensorboard_name
                if config.training.tensorboard_name is not None
                else "lightning_logs"
            ),
        )
        param_log_dict = OmegaConf.to_container(cfg, resolve=True)
        assert is_str_dict(param_log_dict)
        tb_logger.log_hyperparams(param_log_dict)

    # Create Fabric object for DDP
    if config.system.device == "cpu":
        fabric = Fabric(accelerator="cpu")
    elif config.system.device == "cuda":
        fabric = Fabric(
            strategy=DDPStrategy(
                find_unused_parameters=config.system.ddp.find_unused_parameters
            ),
            accelerator="gpu",
            num_nodes=config.system.ddp.nodes,
            devices=config.system.ddp.devices,
            loggers=(
                [
                    tb_logger,
                    # CSVLogger(output_dir + "/csv_logs"),
                ]
                if config.training.log_tensorboard
                else None
            ),
        )
    else:
        raise ValueError(f"Unrecognized device {config.system.device}")

    # Log params
    if fabric.is_global_zero:
        _logger.info("Setting up...")
        _logger.info(print(yaml.dump(OmegaConf.to_container(cfg, resolve=True))))
        with open(os.path.join(output_dir, "config.yaml"), "w") as f:
            OmegaConf.save(cfg, f, True)

    # Launch Fabric
    fabric.seed_everything(config.system.seed + fabric.global_rank)
    fabric.launch()

    # Build Pytorch model
    model = EMModel.from_config(config.model)
    if config.model.backbone.convert_sync_batch_norm and fabric.world_size > 1:
        model = ME.MinkowskiSyncBatchNorm.convert_sync_batchnorm(model)
    if config.training.compile:
        if fabric.is_global_zero:
            _logger.info("torch.compile-ing model...")
        model = torch.compile(model, dynamic=True)

    # load data
    electron_hdf_file = os.path.join(
        config.dataset.directory,
        os.path.splitext(config.dataset.pixels_file)[0] + ".hdf5",
    )
    if not os.path.exists(electron_hdf_file):  # need to convert txt to hdf5
        if fabric.is_global_zero:
            pixels_file = os.path.join(
                config.dataset.directory, config.dataset.pixels_file
            )
            _logger.info(f"Converting {pixels_file} to {electron_hdf_file}...")
            convert_electron_pixel_file_to_hdf5(pixels_file, electron_hdf_file)
            _logger.info("Done converting.")
    fabric.barrier()  # wait for data to be converted if necessary

    # make dataset and dataloader objects
    train_dataset, eval_dataset = make_test_train_datasets(
        electron_hdf_file=electron_hdf_file,
        events_per_image_range=config.dataset.events_per_image_range,
        pixel_patch_size=config.dataset.pixel_patch_size,
        hybrid_sparse_tensors=False,
        train_percentage=config.dataset.train_percentage,
        noise_std=config.dataset.noise_std,
        transform=NSigmaSparsifyTransform(
            config.dataset.n_sigma_sparsify,
            config.dataset.pixel_patch_size,
            max_pixels_to_keep=config.dataset.max_pixels_to_keep,
        ),
        shared_shuffle_seed=config.system.seed,
        new_grid_size=config.dataset.new_grid_size,
    )
    assert train_dataset is not None
    assert eval_dataset is not None
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        config.training.batch_size,
        collate_fn=electron_collate_fn,
        pin_memory=True,
        num_workers=config.dataset.num_workers,
        worker_init_fn=worker_init_fn,
    )
    eval_dataloader = torch.utils.data.DataLoader(
        eval_dataset,
        config.training.batch_size,
        collate_fn=electron_collate_fn,
        pin_memory=True,
        num_workers=config.dataset.num_workers,
        worker_init_fn=worker_init_fn,
    )

    # make optimizer and lr scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.training.lr)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        int(config.training.num_steps * config.training.warmup_percentage),
        config.training.num_steps,
    )

    # give training objects to fabric
    assert isinstance(model, nn.Module)
    model, optimizer = fabric.setup(model, optimizer)
    train_dataloader, eval_dataloader = fabric.setup_dataloaders(
        train_dataloader, eval_dataloader
    )

    # load checkpoint if resuming
    if config.training.resume_file is not None:
        start_iter = load(config.training.resume_file, model, optimizer, fabric)
    else:
        start_iter = 0

    train(
        config,
        fabric,
        model,
        optimizer,
        lr_scheduler,
        train_dataloader,
        eval_dataloader,
        os.path.join(output_dir, "checkpoints"),
        start_iter,
    )


def train(
    config: Config,
    fabric: Fabric,
    model,
    optimizer,
    lr_scheduler,
    train_dataloader,
    eval_dataloader,
    save_dir,
    start_iter: int = 0,
):
    """Main training loop."""
    if config.system.ddp.detect_anomaly:  # debug option
        torch.autograd.set_detect_anomaly(True)

    # iter timer for logging
    iter_timer = RunningMean(config.training.print_interval).to(fabric.device)

    # training setup
    model.train()
    iter_loader = iter(train_dataloader)
    epoch = 0
    criterion: EMCriterion = model.criterion
    training_start_time = time.time()
    if fabric.is_global_zero:
        _logger.info("Begin training.")

    # run for N steps instead of N epochs
    for i in range(start_iter, config.training.num_steps):
        t0 = time.time()
        try:
            batch = next(iter_loader)
        except StopIteration:  # end of epoch
            _logger.info(f"End of epoch {epoch}, beginning evaluation...")
            eval(epoch, i, t0, model, eval_dataloader, fabric)
            model.train()
            epoch += 1
            iter_loader = iter(train_dataloader)
            batch = next(iter_loader)

        # forward pass
        loss_dict, model_output = model(batch)
        assert is_str_dict(loss_dict)
        total_loss = loss_dict["loss"]
        if not torch.isfinite(total_loss):
            raise ValueError(f"Got invalid loss: {total_loss} on step {i}")

        # backward pass
        fabric.backward(total_loss)
        fabric.clip_gradients(model, optimizer, config.training.max_grad_norm)
        if config.system.ddp.find_unused_parameters:
            __debug_find_unused_parameters(model)

        # step everything
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

        # log
        with torch.no_grad():
            log_dict = fabric.all_reduce(loss_dict, reduce_op="mean")
        assert is_str_dict(log_dict)
        log_dict["lr"] = lr_scheduler.get_last_lr()[0]

        # on log interval, average metrics and print them
        if i % config.training.print_interval == 0 and i > 0:
            metric_log_dict = criterion.metric_manager.get_logs("train")
            if criterion.config.use_denoising_loss:
                metric_log_dict.update(criterion.metric_manager.get_logs("denoising"))
            log_str = criterion.metric_manager.make_log_str(metric_log_dict)
            elapsed_time = time.time() - training_start_time
            elapsed_time_str = _elapsed_time_str(elapsed_time)
            iter_time = iter_timer.compute()

            log_str = f"Iter {i} (Epoch {epoch}) -- ({elapsed_time_str}) -- " + log_str
            log_str = log_str + f" iter_time: {iter_time}\n"
            if fabric.is_global_zero:
                _logger.info(log_str)
            metric_log_dict["iter_time"] = iter_time
            log_dict.update(metric_log_dict)

        # log to tensorboard
        log_dict = criterion.metric_manager.format_log_keys(log_dict)
        fabric.log_dict(log_dict, step=i)

        # evaluate every eval_steps steps
        if i > 0 and i % config.training.eval_steps == 0:
            if not config.training.skip_checkpoints:
                save(save_dir, f"step_{i}", model, optimizer, fabric, i)
            _logger.info(f"Beginning evaluation on step {i}...")
            eval(epoch, i, t0, model, eval_dataloader, fabric)
            model.train()
        elif (
            i > 0
            and i % config.training.save_steps == 0
            and not config.training.skip_checkpoints
        ):
            save(save_dir, f"step_{i}", model, optimizer, fabric, i)

        iter_timer.update(time.time() - t0)

        # MinkowskiEngine says to clear the cache periodically
        if i > 0 and i % config.training.clear_cache_interval == 0:
            torch.cuda.empty_cache()

    # all done
    save(save_dir, "final", model, optimizer, fabric, i)
    if fabric.is_global_zero:
        elapsed_time_str = _elapsed_time_str(time.time() - training_start_time)
        _logger.info(f"Training complete in {elapsed_time_str}.")


@torch.no_grad()
def eval(
    epoch: int,
    step: int,
    start_time: float,
    model: nn.Module,
    eval_loader: torch.utils.data.DataLoader,
    fabric: Fabric,
):
    torch.cuda.empty_cache()
    start_eval = time.time()
    model.eval()
    criterion: EMCriterion = model.criterion
    for batch in eval_loader:
        output = model(batch)
        criterion.evaluate_batch(output, batch)
    fabric.barrier()
    metric_log_dict = criterion.metric_manager.get_logs("eval")
    metric_log_dict = criterion.metric_manager.format_log_keys(metric_log_dict)
    log_str = criterion.metric_manager.make_log_str(metric_log_dict)
    elapsed_time = time.time() - start_time
    elapsed_time_str = _elapsed_time_str(elapsed_time)
    eval_time_str = _elapsed_time_str(time.time() - start_eval)
    log_str = (
        f"Evaluation: Iter {step} (Epoch {epoch}) -- ({elapsed_time_str}) -- " + log_str
    )
    log_str = log_str + f"Eval time: {eval_time_str}"
    if fabric.is_global_zero:
        _logger.info(log_str)
    fabric.log_dict(metric_log_dict, step)
    torch.cuda.empty_cache()


def save(
    save_dir: str, save_name: str, model, optimizer, fabric: Fabric, iteration: int
):
    state = {"model": model, "optimizer": optimizer, "iteration": iteration}
    save_file = os.path.join(save_dir, save_name + ".ckpt")
    fabric.save(save_file, state)
    if fabric.is_global_zero:
        _logger.info(f"Saved to {save_file}\n")


def load(checkpoint_file: str, model, optimizer, fabric: Fabric):
    state = {"model": model, "optimizer": optimizer}
    remainder = fabric.load(checkpoint_file, state)
    iteration = remainder["iteration"]
    return iteration


def _elapsed_time_str(elapsed_time):
    return str(datetime.timedelta(seconds=int(elapsed_time)))


def memory_stats(fabric: Fabric) -> dict[str, int]:
    return {
        "allocated": torch.cuda.memory_allocated(fabric.device),
        "cached": torch.cuda.memory_reserved(fabric.device),
        "max_allocated": torch.cuda.max_memory_allocated(fabric.device),
        "max_cached": torch.cuda.max_memory_reserved(fabric.device),
    }


class CudaUsageMonitor(nn.Module):
    MB = 1024**2

    def __init__(self, sample_window: int):
        self.utilization = RunningMean(sample_window)
        self.memory = RunningMean(sample_window)
        self.max_memory = RunningMean(sample_window)

    def update(self):
        self.utilization.update(torch.cuda.utilization(device=self.utilization.device))
        self.memory.update(torch.cuda.memory_allocated(device=self.memory.device))
        self.max_memory.update(
            torch.cuda.max_memory_allocated(device=self.max_memory.device)
        )

    def compute(self):
        utilization = self.utilization.compute()
        memory = self.memory.compute() // self.MB
        max_memory = self.max_memory.compute() // self.max_memory
        return utilization, memory, max_memory


def __debug_find_unused_parameters(model):
    unused_parameters = []
    for name, param in model.named_parameters():
        if param.grad is None:
            unused_parameters.append(name)
    if len(unused_parameters) > 0:
        _logger.debug(f"Unused parameters: {unused_parameters}")


# Doesn't work because can't get each worker's dataset's state
# def verify_data_integrity(dataloader: torch.utils.data.DataLoader, fabric: Fabric):
#     if fabric.world_size <= 1:
#         return
#     dataset: GeantElectronDataset = dataloader.dataset
#     electron_order = torch.tensor(dataset._shuffled_elec_indices, device=fabric.device)
#     rank0_electron_order = fabric.broadcast(electron_order, 0)
#     assert torch.equal(electron_order, rank0_electron_order)

#     thisrank_indices = torch.cat([
#         torch.tensor(chunk, device=fabric.device)
#         for chunk in dataset._chunks_this_loader
#     ])
#     rank0_indices = fabric.broadcast(thisrank_indices, 0)
#     if not fabric.is_global_zero:
#         combined = torch.cat(rank0_indices, thisrank_indices)
#         assert torch.unique(combined).shape[0] == combined.shape[0]
#     fabric.barrier()


if __name__ == "__main__":
    main()
