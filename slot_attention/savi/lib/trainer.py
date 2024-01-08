# Copyright 2022 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""The main model training loop."""

import functools
import os
import time
from typing import Callable, Dict, Iterable, Mapping, Optional, Tuple, Type, Union

from absl import logging
from clu import checkpoint
from clu import metric_writers
from clu import metrics
from clu import parameter_overview
from clu import periodic_actions
import flax
from flax import linen as nn
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
import optax
from savi.lib import evaluator
from savi.lib import input_pipeline
from savi.lib import losses
from savi.lib import utils
import tensorflow as tf

Array = jnp.ndarray
ArrayTree = Union[Array, Iterable["ArrayTree"], Mapping[str, "ArrayTree"]]  # pytype: disable=not-supported-yet
PRNGKey = Array


def train_step(
    model: nn.Module,
    rng: PRNGKey,
    step: int,
    state_vars: flax.core.FrozenDict,
    opt: flax.optim.Optimizer,  # pytype: disable=module-attr
    batch: Dict[str, ArrayTree],
    loss_fn: losses.LossFn,
    learning_rate_fn: Callable[[Array], Array],
    train_metrics_cls: Type[metrics.Collection],
    predicted_max_num_instances: int,
    ground_truth_max_num_instances: int,
    conditioning_key: Optional[str] = None,
    max_grad_norm: Optional[float] = None,
    ) -> Tuple[flax.optim.Optimizer, flax.core.FrozenDict, PRNGKey,  # pytype: disable=module-attr
               metrics.Collection, int]:
  """Perform a single training step.

  Args:
    model: Model used in training step.
    rng: Random number key
    step: Which training step we are on.
    state_vars: Accessory variables.
    opt: The optimizer to use to minimize loss_fn.
    batch: Training inputs for this step.
    loss_fn: Loss function that takes model predictions and a batch of data.
    learning_rate_fn: Function that outputs learning rate as jnp.float32 given
      step as jnp.int*.
    train_metrics_cls: The metrics collection for computing training metrics.
    predicted_max_num_instances: Maximum number of instances in prediction.
    ground_truth_max_num_instances: Maximum number of instances in ground truth,
      including background (which counts as a separate instance).
    conditioning_key: Optional string. If provided, defines the batch key to be
      used as conditioning signal for the model. Otherwise this is inferred from
      the available keys in the batch.
    max_grad_norm: Optional float, if not None, clip gradients to the specified
      maximum norm.

  Returns:
    Tuple of the updated opt, state_vars, new random number key,
      metrics update, and step + 1. Note that some of this info is stored in
      TrainState, but here it is unpacked.
  """

  # Split PRNGKey and bind to host / device.
  new_rng, rng = jax.random.split(rng)
  rng = jax.random.fold_in(rng, jax.host_id())
  rng = jax.random.fold_in(rng, jax.lax.axis_index("batch"))
  init_rng, dropout_rng = jax.random.split(rng, 2)

  mutable_var_keys = list(state_vars.keys()) + ["intermediates"]

  conditioning = batch[conditioning_key] if conditioning_key else None

  def train_loss_fn(params, state_vars):
    preds, mutable_vars = model.apply(
        {"params": params, **state_vars}, video=batch["video"],
        conditioning=conditioning, mutable=mutable_var_keys,
        rngs={"state_init": init_rng, "dropout": dropout_rng}, train=True,
        padding_mask=batch.get("padding_mask"))
    # Filter intermediates, as we do not want to store them in the TrainState.
    state_vars = utils.filter_key_from_frozen_dict(
        mutable_vars, key="intermediates")
    loss, loss_aux = loss_fn(preds, batch)
    return loss, (state_vars, preds, loss_aux)

  grad_fn = jax.value_and_grad(train_loss_fn, has_aux=True)
  (loss, (state_vars, preds, loss_aux)), grad = grad_fn(opt.target, state_vars)

  # Compute average gradient across multiple workers.
  grad = jax.lax.pmean(grad, axis_name="batch")

  if max_grad_norm is not None:
    grad = utils.clip_grads(grad, max_grad_norm)

  # Subtracting 1 from step as we start from initial step 1 instead of 0.
  learning_rate = learning_rate_fn(step - 1)
  opt = opt.apply_gradient(grad, learning_rate=learning_rate)

  # Compute metrics.
  metrics_update = train_metrics_cls.gather_from_model_output(
      loss=loss,
      **loss_aux,
      predicted_segmentations=utils.remove_singleton_dim(
          preds["outputs"].get("segmentations")),  # pytype: disable=attribute-error
      ground_truth_segmentations=batch.get("segmentations"),
      predicted_max_num_instances=predicted_max_num_instances,
      ground_truth_max_num_instances=ground_truth_max_num_instances,
      padding_mask=batch.get("padding_mask"),
      mask=batch.get("mask"))
  return opt, state_vars, new_rng, metrics_update, step + 1


def train_and_evaluate(config: ml_collections.ConfigDict,
                       workdir: str):
  """Runs a training and evaluation loop.

  Args:
    config: Configuration to use.
    workdir: Working directory for checkpoints and TF summaries. If this
      contains checkpoint training will be resumed from the latest checkpoint.
  """
  rng = jax.random.PRNGKey(config.seed)

  tf.io.gfile.makedirs(workdir)

  # Input pipeline.
  rng, data_rng = jax.random.split(rng)
  # Make sure each host uses a different RNG for the training data.
  if config.get("seed_data", True):  # Default to seeding data if not specified.
    data_rng = jax.random.fold_in(data_rng, jax.host_id())
  else:
    data_rng = None
  train_ds, eval_ds = input_pipeline.create_datasets(config, data_rng)
  train_iter = iter(train_ds)  # pytype: disable=wrong-arg-types

  # Initialize model
  model = utils.build_model_from_config(config.model)
  learning_rate_fn = optax.warmup_cosine_decay_schedule(
      init_value=0.,
      peak_value=config.learning_rate,
      warmup_steps=config.warmup_steps,
      decay_steps=config.num_train_steps)
  optimizer_def = flax.optim.Adam(learning_rate=config.learning_rate)  # pytype: disable=module-attr

  # Construct TrainMetrics and EvalMetrics, metrics collections.
  train_metrics_cls = utils.make_metrics_collection("TrainMetrics",
                                                    config.train_metrics_spec)
  eval_metrics_cls = utils.make_metrics_collection("EvalMetrics",
                                                   config.eval_metrics_spec)

  def init_model(rng):
    rng, init_rng, model_rng, dropout_rng = jax.random.split(rng, num=4)

    init_conditioning = None
    if config.get("conditioning_key"):
      init_conditioning = jnp.ones(
          [1] + list(train_ds.element_spec[config.conditioning_key].shape)[2:],
          jnp.int32)
    init_inputs = jnp.ones(
        [1] + list(train_ds.element_spec["video"].shape)[2:],
        jnp.float32)
    initial_vars = model.init(
        {"params": model_rng, "state_init": init_rng, "dropout": dropout_rng},
        video=init_inputs, conditioning=init_conditioning,
        padding_mask=jnp.ones(init_inputs.shape[:-1], jnp.int32))

    # Split into state variables (e.g. for batchnorm stats) and model params.
    # Note that `pop()` on a FrozenDict performs a deep copy.
    state_vars, initial_params = initial_vars.pop("params")  # pytype: disable=attribute-error

    # Filter out intermediates (we don't want to store these in the TrainState).
    state_vars = utils.filter_key_from_frozen_dict(
        state_vars, key="intermediates")
    return state_vars, initial_params

  state_vars, initial_params = init_model(rng)
  parameter_overview.log_parameter_overview(initial_params)  # pytype: disable=wrong-arg-types
  optimizer = optimizer_def.create(initial_params)

  state = utils.TrainState(
      step=1, optimizer=optimizer, rng=rng, variables=state_vars)

  loss_fn = functools.partial(
      losses.compute_full_loss, loss_config=config.losses)

  checkpoint_dir = os.path.join(workdir, "checkpoints")
  ckpt = checkpoint.MultihostCheckpoint(checkpoint_dir)
  state = ckpt.restore_or_initialize(state)
  initial_step = int(state.step)

  # Replicate our parameters.
  state = flax.jax_utils.replicate(state, devices=jax.local_devices())
  del rng  # rng is stored in the state.

  # Only write metrics on host 0, write to logs on all other hosts.
  writer = metric_writers.create_default_writer(
      workdir, just_logging=jax.host_id() > 0)
  writer.write_hparams(utils.prepare_dict_for_logging(config.to_dict()))

  logging.info("Starting training loop at step %d.", initial_step)
  report_progress = periodic_actions.ReportProgress(
      num_train_steps=config.num_train_steps, writer=writer)
  if jax.process_index() == 0:
    profiler = periodic_actions.Profile(num_profile_steps=5, logdir=workdir)
  p_train_step = jax.pmap(
      train_step,
      axis_name="batch",
      donate_argnums=(1, 2, 3, 4, 5),
      static_broadcasted_argnums=(0, 6, 7, 8, 9, 10, 11, 12))

  train_metrics = None
  with metric_writers.ensure_flushes(writer):
    if config.num_train_steps == 0:
      with report_progress.timed("eval"):
        evaluate(model, state, eval_ds, loss_fn, eval_metrics_cls, config,
                 writer, step=0)
      with report_progress.timed("checkpoint"):
        ckpt.save(flax.jax_utils.unreplicate(state))
      return

    for step in range(initial_step, config.num_train_steps + 1):
      # `step` is a Python integer. `state.step` is JAX integer on GPU/TPU.
      is_last_step = step == config.num_train_steps

      with jax.profiler.StepTraceAnnotation("train", step_num=step):
        batch = jax.tree_map(np.asarray, next(train_iter))
        opt, state_vars, rng, metrics_update, p_step = p_train_step(
            model, state.rng, state.step, state.variables,
            state.optimizer, batch, loss_fn, learning_rate_fn,
            train_metrics_cls,
            config.num_slots,
            config.max_instances + 1,  # Incl. background.
            config.get("conditioning_key"),
            config.get("max_grad_norm"))

        state = state.replace(  # pytype: disable=attribute-error
            optimizer=opt,
            step=p_step,
            variables=state_vars,
            rng=rng,
        )

        metric_update = flax.jax_utils.unreplicate(metrics_update)
        train_metrics = (
            metric_update
            if train_metrics is None else train_metrics.merge(metric_update))

      # Quick indication that training is happening.
      logging.log_first_n(logging.INFO, "Finished training step %d.", 5, step)
      report_progress(step, time.time())

      if jax.process_index() == 0:
        profiler(step)

      if step % config.log_loss_every_steps == 0 or is_last_step:
        metrics_res = train_metrics.compute()
        writer.write_scalars(step, jax.tree_map(np.array, metrics_res))
        train_metrics = None

      if step % config.eval_every_steps == 0 or is_last_step:
        with report_progress.timed("eval"):
          evaluate(model, state, eval_ds, loss_fn, eval_metrics_cls,
                   config, writer, step=step)

      if step % config.checkpoint_every_steps == 0 or is_last_step:
        with report_progress.timed("checkpoint"):
          ckpt.save(flax.jax_utils.unreplicate(state))


def evaluate(model, state, eval_ds, loss_fn_eval, eval_metrics_cls, config,
             writer, step: int):
  """Evaluate the model."""
  eval_metrics, eval_batch, eval_preds = evaluator.evaluate(
      model,
      state,
      eval_ds,
      loss_fn_eval,
      eval_metrics_cls,
      predicted_max_num_instances=config.num_slots,
      ground_truth_max_num_instances=config.max_instances + 1,  # Incl. bg.
      slice_size=config.get("eval_slice_size"),
      slice_keys=config.get("eval_slice_keys"),
      conditioning_key=config.get("conditioning_key"),
      remove_from_predictions=config.get("remove_from_predictions"),
      metrics_on_cpu=config.get("metrics_on_cpu", False))

  metrics_res = eval_metrics.compute()
  writer.write_scalars(
      step, jax.tree_map(np.array, utils.flatten_named_dicttree(metrics_res)))
  writer.write_images(
      step,
      jax.tree_map(
          np.array,
          utils.prepare_images_for_logging(
              config,
              eval_batch,
              eval_preds,
              n_samples=config.get("n_samples", 5),
              n_frames=config.get("n_frames", 5),
              min_n_colors=config.get("logging_min_n_colors", 1))))
