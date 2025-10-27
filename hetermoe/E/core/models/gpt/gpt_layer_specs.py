# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add
from megatron.core.fusions.fused_layer_norm import FusedLayerNorm
from megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear
from megatron.core.transformer.attention import SelfAttention, SelfAttentionSubmodules
from megatron.core.transformer.custom_layers.transformer_engine import (
    TEDotProductAttention,
    TELayerNormColumnParallelLinear,
    TENorm,
    TERowParallelLinear,
)
from megatron.core.transformer.dot_product_attention import DotProductAttention
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.mlp import MLP, MLPSubmodules
from megatron.core.transformer.moe.moe_layer import MoELayer
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_block import TransformerBlockSubmodules
from megatron.core.transformer.transformer_layer import TransformerLayer, TransformerLayerSubmodules
from hetermoe.E.core.transformer.transformer_layer import HeterExpertTransformerLayer,HeterExpertTransformerLayerSubmodules
from hetermoe.E.core.transformer.moe.moe_layer import HeterMoELayer
from megatron.training.global_vars import get_args


     

def get_gpt_layer_local_spec_heter_moe_is_E(
    num_experts: int = None, moe_grouped_gemm: bool = False
)-> ModuleSpec:
    mlp = _get_mlp_module_spec(
        use_te=False, num_experts=num_experts, moe_grouped_gemm=moe_grouped_gemm
    )

    return ModuleSpec(
        module=HeterExpertTransformerLayer,
        submodules=HeterExpertTransformerLayerSubmodules(
            mlp=mlp,
            # sharded_state_dict_keys_map={
            #     'input_layernorm.': 'self_attention.linear_qkv.layer_norm_',
            #     'pre_mlp_layernorm.': 'mlp.linear_fc1.layer_norm_',
            # },
        ),
    )



    

# Helper function to get module spec for MLP/MoE
def _get_mlp_module_spec(
    use_te: bool = True, num_experts: int = None, moe_grouped_gemm: bool = False
) -> ModuleSpec:
    if num_experts is None:
        # Dense MLP w/ or w/o TE modules.
        return ModuleSpec(
            module=HeterMoELayer,
            submodules=MLPSubmodules(linear_fc1=ColumnParallelLinear, linear_fc2=RowParallelLinear,)
            if not moe_grouped_gemm
            else None,
        )




    

