import os
from pathlib import Path
import logging
from scripts import losses
from scripts import sampling
from scripts.sampling_bd import get_sampling_fn_blurring_diffusion
from model_code import utils as mutils
from model_code.ema import ExponentialMovingAverage
from scripts import datasets
import torch
from torch.utils import tensorboard
from scripts import utils
from absl import app
from absl import flags
from ml_collections.config_flags import config_flags
from model_code.blurring_diffusion import BlurringDiffusion

FLAGS = flags.FLAGS

config_flags.DEFINE_config_file(
    "config", None, "Training configuration.", lock_config=True)
flags.DEFINE_string("workdir", None, "Work directory.")
flags.mark_flags_as_required(["workdir", "config"])
#flags.DEFINE_string("initialization", "prior", "How to initialize sampling")


def main(argv):
    train(FLAGS.config, FLAGS.workdir)




def get_step_fn(train, bd, config, optimize_fn=None,device=None):
    """A wrapper for loss functions in training or evaluation
    Based on code from https://github.com/yang-song/score_sde_pytorch"""
    if device == None:
        device = config.device

    loss_fn = bd.loss

    # For automatic mixed precision
    scaler = torch.cuda.amp.GradScaler()

    def step_fn(state, batch):
        """Running one step of training or evaluation.
        Returns:
                loss: The average loss value of this state.
        """
        model = state['model']
        if train:
            optimizer = state['optimizer']
            if config.optim.automatic_mp:
                optimizer.zero_grad()
                with torch.cuda.amp.autocast():
                    loss, losses_batch, fwd_steps_batch = loss_fn(model, x = batch)
                    # amp not recommended in backward pass, but had issues getting this to work without it
                    # Followed https://github.com/pytorch/pytorch/issues/37730
                    scaler.scale(loss).backward()
                scaler.scale(losses_batch)
                optimize_fn(optimizer, model.parameters(), step=state['step'],
                            scaler=scaler)
                state['step'] += 1
                state['ema'].update(model.parameters())
            else:
                optimizer.zero_grad()
                loss, losses_batch, fwd_steps_batch = loss_fn(model, batch)
                loss.backward()
                optimize_fn(optimizer, model.parameters(), step=state['step'])
                state['step'] += 1
                state['ema'].update(model.parameters())
        else:
            with torch.no_grad():
                ema = state['ema']
                ema.store(model.parameters())
                ema.copy_to(model.parameters())
                loss, losses_batch, fwd_steps_batch = loss_fn(model, batch)
                ema.restore(model.parameters())

        return loss, losses_batch, fwd_steps_batch

    return step_fn


def train(config, workdir):
    """Runs the training pipeline. 
    Based on code from https://github.com/yang-song/score_sde_pytorch

    Args:
            config: Configuration to use.
            workdir: Working directory for checkpoints and TF summaries. If this
                    contains checkpoint training will be resumed from the latest checkpoint.
    """

    if config.device == torch.device('cpu'):
        logging.info("RUNNING ON CPU")

    # Create directory for saving intermediate samples
    sample_dir = os.path.join(workdir, "samples")
    Path(sample_dir).mkdir(parents=True, exist_ok=True)
    # Create directory for tensorboard logs
    tb_dir = os.path.join(workdir, "tensorboard")
    Path(tb_dir).mkdir(parents=True, exist_ok=True)
    writer = tensorboard.SummaryWriter(tb_dir)

    # Initialize model
    model = mutils.create_model(config)
    optimizer = losses.get_optimizer(config, model.parameters())
    ema = ExponentialMovingAverage(
        model.parameters(), decay=config.model.ema_rate)
    state = dict(optimizer=optimizer, model=model, step=0, ema=ema)
    model_evaluation_fn = mutils.get_model_fn(model, train=False)

    # Create checkpoints directory
    checkpoint_dir = os.path.join(workdir, "checkpoints")
    # Intermediate checkpoints to resume training
    checkpoint_meta_dir = os.path.join(
        workdir, "checkpoints-meta", "checkpoint.pth")
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(os.path.dirname(checkpoint_meta_dir)).mkdir(
        parents=True, exist_ok=True)
    # Resume training when intermediate checkpoints are detected
    state = utils.restore_checkpoint(checkpoint_meta_dir, state, config.device)
    initial_step = int(state['step'])

    # Build data iterators
    trainloader, testloader = datasets.get_dataset(
        config, uniform_dequantization=config.data.uniform_dequantization)
    train_iter = iter(trainloader)
    eval_iter = iter(testloader)

    # Build one-step training and evaluation functions
    optimize_fn = losses.optimization_manager(config)

    # Get the forward process definition
    bd = BlurringDiffusion(
        T=config.model.T, 
        res=config.data.image_size, 
        sigma_blur_max=config.model.blur_sigma_max, 
        min_scale=config.model.min_scale, 
        logsnr_min=config.model.logsnr_min, 
        logsnr_max=config.model.logsnr_max,
        delta=config.model.delta
        )


    # Get the loss function
    train_step_fn = get_step_fn(train=True, bd=bd, config=config, optimize_fn=optimize_fn)
    eval_step_fn = get_step_fn(train=False, bd=bd, config=config, optimize_fn=optimize_fn)
    

    # Building sampling functions
    initial_noise = torch.randn(config.training.batch_size, 3, config.data.image_size, config.data.image_size, dtype=torch.float)
    sampling_fn = get_sampling_fn_blurring_diffusion(config = config, 
                                                     initial_noise=initial_noise,
                                                     denoise_fn = bd.denoise,
                                                     intermediate_sample_indices = list(range(config.model.T+1)), 
                                                     device=config.device)

    num_train_steps = config.training.n_iters
    logging.info("Starting training loop at step %d." % (initial_step,))
    logging.info("Running on {}".format(config.device))

    # # For analyzing the mean values of losses over many batches, for each scale separately
    # pooled_losses = torch.zeros(len(scales))

    for step in range(initial_step, num_train_steps + 1):
        # Train step
        try:
            batch = next(train_iter)[0].to(config.device).float()
        except StopIteration:  # Start new epoch if run out of data
            train_iter = iter(trainloader)
            batch = next(train_iter)[0].to(config.device).float()
        loss, losses_batch, fwd_steps_batch = train_step_fn(state, batch)

        writer.add_scalar("training_loss", loss.item(), step)

        # Save a temporary checkpoint to resume training if training is stopped
        if step != 0 and step % config.training.snapshot_freq_for_preemption == 0:
            logging.info("Saving temporary checkpoint")
            utils.save_checkpoint(checkpoint_meta_dir, state)

        # Report the loss on an evaluation dataset periodically
        if step % config.training.eval_freq == 0:
            logging.info("Starting evaluation")
            # Use 25 batches for test-set evaluation, arbitrary choice
            N_evals = 25
            for i in range(N_evals):
                try:
                    eval_batch = next(eval_iter)[0].to(config.device).float()
                except StopIteration:  # Start new epoch
                    eval_iter = iter(testloader)
                    eval_batch = next(eval_iter)[0].to(config.device).float()
                eval_loss, losses_batch, fwd_steps_batch = eval_step_fn(state, eval_batch)
                eval_loss = eval_loss.detach()
            logging.info("step: %d, eval_loss: %.5e" % (step, eval_loss.item()))

        # Save a checkpoint periodically
        if step != 0 and step % config.training.snapshot_freq == 0 or step == num_train_steps:
            logging.info("Saving a checkpoint")
            # Save the checkpoint.
            save_step = step // config.training.snapshot_freq
            utils.save_checkpoint(os.path.join(
                checkpoint_dir, 'checkpoint_{}.pth'.format(save_step)), state)

        # Generate samples periodically
        if step != 0 and step % config.training.sampling_freq == 0 or step == num_train_steps:
            logging.info("Sampling...")
            ema.store(model.parameters())
            ema.copy_to(model.parameters())
            sample, n, intermediate_samples = sampling_fn(model_evaluation_fn)
            ema.restore(model.parameters())
            this_sample_dir = os.path.join(sample_dir, "iter_{}".format(step))
            Path(this_sample_dir).mkdir(parents=True, exist_ok=True)
            utils.save_tensor(this_sample_dir, sample, "final.np")
            utils.save_png(this_sample_dir, sample, "final.png")
            utils.save_gif(this_sample_dir, intermediate_samples)
            utils.save_video(this_sample_dir, intermediate_samples)


if __name__ == "__main__":
    app.run(main)
