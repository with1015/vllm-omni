from collections.abc import Iterable, Mapping

from vllm.model_executor.models.hcx_omni import (
    HCXOmniForCausalLM,
    HCXOmniDummyInputsBuilder,
    HCXOmniMultiModalProcessor,
    HCXOmniProcessingInfo,
)

from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.parse import MultiModalDataItems
from vllm.multimodal.inputs import MultiModalKwargsItems
from vllm.multimodal.processing.processor import (
    MultiModalPromptUpdates,
    PlaceholderFeaturesInfo,
)

from vllm.model_executor.models.utils import (
    maybe_prefix,
    init_vllm_registered_model
)

from vllm.model_executor.models.interfaces import (
    MultiModalEmbeddings,
    SupportsMultiModal,
    SupportsPP,
)

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.sequence import IntermediateTensors

from vllm_omni.model_executor.models.utils import add_prefix_to_loaded_weights
from vllm_omni.model_executor.models.output_templates import OmniOutput

import torch
import torch.nn as nn

logger = init_logger(__name__)


@MULTIMODAL_REGISTRY.register_processor(
    HCXOmniMultiModalProcessor,
    info=HCXOmniProcessingInfo,
    dummy_inputs=HCXOmniDummyInputsBuilder,
)
class HyperCLOVAXOmniForCausalLM(nn.Module, SupportsMultiModal, SupportsPP):
    """Wrapper so OmniModelRegistry loads this class; multimodal processor comes from base."""

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        # Delegate placeholder formatting (e.g., <|VIDEO_PAD|>) to the
        # underlying vLLM HyperCLOVAX Omni implementation.
        return HCXOmniForCausalLM.get_placeholder_str(modality, i)

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        self.config = vllm_config.model_config.hf_config
        self.model_stage = vllm_config.model_config.model_stage
        self.multimodal_config = vllm_config.model_config.multimodal_config

        if self.model_stage == "thinker":
            hcx_config = vllm_config.with_hf_config(
                self.config, architectures=["HCXOmniForCausalLM"]
            )
            self.hcx_omni = init_vllm_registered_model(
                vllm_config=hcx_config,
                prefix=maybe_prefix(prefix, "hyperclovax_omni"),
                architectures=["HCXOmniForCausalLM"],
            )
        elif self.model_stage == "decoder/audio":
            pass
        else:
            raise ValueError(f"Invalid model stage: {self.model_stage}")

    @staticmethod
    def _module_device(module: nn.Module) -> torch.device:
        """Get the device of a module."""
        try:
            return next(module.parameters()).device
        except StopIteration:
            # No parameters; fall back to buffers or cpu
            for _, buf in module.named_buffers(recurse=True):
                return buf.device
            return torch.device("cpu")

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor | IntermediateTensors | OmniOutput:
        if self.model_stage == "thinker":
            """Forward pass delegates to the base HyperCLOVAX Omni model."""
            output = self.hcx_omni.forward(
                input_ids,
                positions,
                intermediate_tensors=intermediate_tensors,
                inputs_embeds=inputs_embeds,
                **kwargs,
            )
            return output

    def compute_logits(self, hidden_states: torch.Tensor, **kwargs: object) -> torch.Tensor | None:
        return self.hcx_omni.compute_logits(hidden_states, **kwargs)

    #def get_mm_mapping(self):
    #    return self.hcx_omni.get_mm_mapping()

    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings:
        """Delegate to base so the runner gets encoder outputs for inputs_embeds."""
        return self.hcx_omni.embed_multimodal(**kwargs)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loaded_weights = set()
        hcx_loaded = self.hcx_omni.load_weights(weights)
        # vLLM strict weight init check compares `loaded_weights` with
        # this wrapper's `model.named_parameters()`.
        # Since we delegate to `self.hcx_omni.load_weights()`, we need to
        # prefix the returned names with the attribute path under this wrapper.
        hcx_loaded = add_prefix_to_loaded_weights(hcx_loaded, "hcx_omni")
        loaded_weights.update(hcx_loaded)

        logger.info(
            "Loaded %d weights for HyperCLOVAXOmni (stage=%s)",
            len(loaded_weights),
            self.model_stage,
        )

        return loaded_weights