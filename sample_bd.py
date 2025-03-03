import os
from pathlib import Path
import logging
from scripts import sampling
from model_code import utils as mutils
import torch
from scripts import utils
from absl import app
from absl import flags
from ml_collections.config_flags import config_flags
from scripts.sampling_bd import get_sampling_fn_blurring_diffusion
from model_code.blurring_diffusion import BlurringDiffusion


FLAGS = flags.FLAGS

config_flags.DEFINE_config_file(
    "config", None, "Training configuration.", lock_config=True)
flags.DEFINE_string("workdir", None, "Work directory.")
flags.DEFINE_integer("checkpoint", None,
                     "Checkpoint number to use for custom sampling")
flags.mark_flags_as_required(["workdir", "config", "checkpoint"])
flags.DEFINE_integer("save_sample_freq", 1,
                     "How often to save samples for output videos?")
flags.DEFINE_float(
    "delta", 0.01, "The standard deviation of noise to add at each step with predicted reverse blur")
flags.DEFINE_integer(
    "batch_size", None, "Batch size of sampled images. Defaults to the training batch size")
flags.DEFINE_bool("same_init", False,
                  "Whether to initialize all samples at the same image")
flags.DEFINE_bool("share_noise", False,
                  "Whether to use the same noises for each image in the generated batch")
flags.DEFINE_integer(
    "num_points", 10, "Default amount of points for sweeping the input from one place to another")
flags.DEFINE_float("final_noise", None,
                   "How much should the noise at the end be? Linear interpolation from noise_amount ot this. If none, use noise_amount")
flags.DEFINE_bool("interpolate", False, "Whether to do interpolation")
flags.DEFINE_integer(
    "number", None, "add a number suffix to generated sample in interpolate")


def main(argv):
    if FLAGS.interpolate:
        raise NotImplementedError
        # sample_interpolate(FLAGS.config, FLAGS.workdir, FLAGS.checkpoint,
        #                    FLAGS.delta, FLAGS.num_points, FLAGS.number)
    else:
        sample(FLAGS.config, FLAGS.workdir, FLAGS.checkpoint, FLAGS.save_sample_freq, FLAGS.delta,
               FLAGS.batch_size, FLAGS.share_noise, FLAGS.same_init)


def sample(config, workdir, checkpoint, save_sample_freq=1,
           delta=None, batch_size=None, share_noise=False, same_init=False):

    if batch_size == None:
        batch_size = config.training.batch_size

    if checkpoint > 0:
        checkpoint_dir = os.path.join(workdir, "checkpoints")
        model = utils.load_model_from_checkpoint(
            config, checkpoint_dir, checkpoint)
    else:  # Checkpoint means the latest checkpoint
        checkpoint_dir = os.path.join(workdir, "checkpoints-meta")
        model = utils.load_model_from_checkpoint_dir(config, checkpoint_dir)

    model_fn = mutils.get_model_fn(model, train=False)
    logging.info("Loaded model from {}".format(checkpoint_dir))
    logging.info("Running on {}".format(config.device))

    logging.info("Creating the forward process...")

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
    logging.info("Done")
    initial_noise = torch.randn(config.training.batch_size, 3, config.data.image_size, config.data.image_size, dtype=torch.float)
    
    if same_init:
        initial_noise = torch.cat(batch_size*[initial_noise[0][None]], 0)
    initial_noise = initial_noise[:batch_size]
    sampling_shape = initial_noise.shape

    intermediate_sample_indices = list(
        range(0, config.model.T+1, save_sample_freq))
    sample_dir = os.path.join(workdir, "additional_samples")
    this_sample_dir = os.path.join(
        sample_dir, "checkpoint_{}".format(checkpoint))

    # Get smapling function and save directory
    sampling_fn = get_sampling_fn_blurring_diffusion(config = config, 
                                                    initial_noise=initial_noise,
                                                    denoise_fn = bd.denoise,
                                                    intermediate_sample_indices = intermediate_sample_indices, 
                                                    device=config.device)
    this_sample_dir = os.path.join(this_sample_dir, "delta_{}".format(delta))
    if same_init:
        this_sample_dir += "_same_init"
    if share_noise:
        this_sample_dir += "_share_noise"

    Path(this_sample_dir).mkdir(parents=True, exist_ok=True)

    logging.info("Do sampling")
    sample, n, intermediate_samples = sampling_fn(model_fn)

    # Save results
    utils.save_tensor_list(this_sample_dir, intermediate_samples, "samples.np")
    utils.save_tensor(this_sample_dir, sample, "final.np")
    utils.save_png(this_sample_dir, sample, "final.png")
    utils.save_png(this_sample_dir, initial_noise, "init.png")
    utils.save_gif(this_sample_dir, intermediate_samples)
    utils.save_video(this_sample_dir, intermediate_samples)



if __name__ == "__main__":
    app.run(main)
