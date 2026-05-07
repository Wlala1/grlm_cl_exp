"""Training with sliding-window routing + aux loss.

Usage:
    python -m routing.train_with_routing \
        --model_name_or_path models/Qwen3-0.6B \
        --dataset grlm_indomain_books_cl_D0 \
        --sliding_window 512 \
        --full_layer 14 \
        --aux_loss_weight 0.1 \
        --output_dir checkpoints/routed_06b_D0 \
        [... other LlamaFactory SFT args ...]

This script:
1. Patches the model config to enable sliding window on all layers except one.
2. Loads the model via LlamaFactory's normal pipeline.
3. Attaches an aux prediction head on the full-attention layer.
4. Monkey-patches the trainer's compute_loss to include aux loss.
5. Runs training.
"""

import sys
import os
from dataclasses import dataclass, field

from transformers import HfArgumentParser

# Add LlamaFactory to path
WORK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LLAMA_DIR = os.path.join(WORK_DIR, "LlamaFactory")
if os.path.isdir(os.path.join(LLAMA_DIR, "src")):
    sys.path.insert(0, os.path.join(LLAMA_DIR, "src"))

from llamafactory.hparams import (
    DataArguments,
    FinetuningArguments,
    GeneratingArguments,
    ModelArguments,
    TrainingArguments,
)
from llamafactory.train.sft.trainer import CustomSeq2SeqTrainer
from llamafactory.train.sft.workflow import run_sft
from llamafactory.model import load_model, load_tokenizer
from transformers import AutoConfig

from routing.config_patch import patch_qwen3_config_for_routing
from routing.aux_head import enable_aux_head


@dataclass
class RoutingArguments:
    """Arguments for sliding window routing."""

    sliding_window: int = field(
        default=512,
        metadata={"help": "Sliding window size in tokens. ~21 tokens per item, so 512 ≈ 24 items."},
    )
    full_layer: int | None = field(
        default=None,
        metadata={"help": "Layer index for full attention. Default: middle layer."},
    )
    aux_loss_weight: float = field(
        default=0.1,
        metadata={"help": "Weight for the auxiliary prediction loss. 0 = disable aux head."},
    )


def main():
    parser = HfArgumentParser((
        ModelArguments,
        DataArguments,
        TrainingArguments,
        FinetuningArguments,
        GeneratingArguments,
        RoutingArguments,
    ))

    (
        model_args,
        data_args,
        training_args,
        finetuning_args,
        generating_args,
        routing_args,
    ) = parser.parse_args_into_dataclasses()

    # Step 1: Determine full_layer index
    config = AutoConfig.from_pretrained(model_args.model_name_or_path)
    full_layer = routing_args.full_layer
    if full_layer is None:
        full_layer = config.num_hidden_layers // 2

    print(f"[routing] sliding_window={routing_args.sliding_window}, "
          f"full_layer={full_layer}/{config.num_hidden_layers}, "
          f"aux_weight={routing_args.aux_loss_weight}")

    # Monkey-patch LlamaFactory's patch_config to inject routing config
    from llamafactory.model import patcher as _patcher
    _original_patch_config = _patcher.patch_config
    sw_size = routing_args.sliding_window

    def _patch_config_with_routing(config, tokenizer, model_args, init_kwargs, is_trainable):
        _original_patch_config(config, tokenizer, model_args, init_kwargs, is_trainable)
        patch_qwen3_config_for_routing(config, sliding_window_size=sw_size, full_layer=full_layer)

    _patcher.patch_config = _patch_config_with_routing
    # Also patch the import in loader module
    from llamafactory.model import loader as _loader_module
    _loader_module.patch_config = _patch_config_with_routing

    # Step 2: Monkey-patch the trainer's compute_loss to add aux loss
    aux_weight = routing_args.aux_loss_weight
    _original_compute_loss = CustomSeq2SeqTrainer.compute_loss

    def _compute_loss_with_aux(self, model, inputs, *args, **kwargs):
        if hasattr(model, "_aux_hidden"):
            model._aux_hidden = None

        loss = _original_compute_loss(self, model, inputs, *args, **kwargs)

        if aux_weight > 0 and hasattr(model, "_aux_head") and model._aux_hidden is not None:
            labels = inputs.get("labels")
            if labels is not None:
                aux_loss = model._aux_head(model._aux_hidden, labels)
                loss = loss + aux_weight * aux_loss

        return loss

    CustomSeq2SeqTrainer.compute_loss = _compute_loss_with_aux

    # Step 3: Monkey-patch load_model to attach aux head after loading
    from llamafactory.model import loader as _loader
    _original_load_model = _loader.load_model

    def _load_model_with_aux(tokenizer, model_args, finetuning_args, is_trainable=False, **kwargs):
        model = _original_load_model(tokenizer, model_args, finetuning_args, is_trainable, **kwargs)
        if is_trainable and aux_weight > 0:
            enable_aux_head(model, full_layer)
            print(f"[routing] aux head attached after layer {full_layer}")
        return model

    _loader.load_model = _load_model_with_aux
    # Also patch the reference in the workflow module
    import llamafactory.train.sft.workflow as _workflow
    _workflow.load_model = _load_model_with_aux

    # Step 4: Run standard LlamaFactory SFT
    run_sft(model_args, data_args, training_args, finetuning_args, generating_args)


if __name__ == "__main__":
    main()
