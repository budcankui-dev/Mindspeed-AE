# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

from abc import ABC, abstractmethod

import torch

from megatron.core import parallel_state
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.moe.router import TopKRouter
from megatron.core.transformer.moe.token_dispatcher import (
    MoEAllGatherTokenDispatcher,
    MoEAlltoAllTokenDispatcher,
)
from megatron.core.transformer.transformer_config import TransformerConfig

import types
from copy import deepcopy
from functools import wraps
import torch.nn.functional as F
from mindspeed.moe.utils import MoEAuxLossAutoScaler
from megatron.core.tensor_parallel import ColumnParallelLinear, RowParallelLinear
from megatron.core.transformer import build_module
from megatron.core.transformer.mlp import MLPSubmodules, MLP
from megatron.core.transformer.moe.experts import GroupedMLP, SequentialMLP
from megatron.core.transformer.moe.moe_utils import save_to_aux_losses_tracker
from megatron.training import get_args




class BaseMoELayer(MegatronModule, ABC):
    """Base class for a mixture of experts layer.

    Args:
        config (TransformerConfig): Configuration object for the transformer model.
    """

    def __init__(self, config: TransformerConfig, layer_number: int = None):
        super(BaseMoELayer, self).__init__(config)
        self.config = config
        # self.expert_parallel_size = parallel_state.get_expert_model_parallel_world_size()
        # assert self.expert_parallel_size > 0, "Expected non-negative expert parallel size"
        # assert self.config.num_moe_experts % self.expert_parallel_size == 0
        self.expert_parallel_size = 1
        self.num_local_experts = self.config.num_moe_experts // self.expert_parallel_size

        # local_expert_indices_offset = (
        #     parallel_state.get_expert_model_parallel_rank() * self.num_local_experts
        # )
        local_expert_indices_offset = 0

        self.local_expert_indices = [
            local_expert_indices_offset + i for i in range(self.num_local_experts)
        ]
        assert all(map(lambda x: x < self.config.num_moe_experts, self.local_expert_indices))
        self.router = None
        self.experts = None
        self.token_dispatcher = None
        self.layer_number = layer_number

    @abstractmethod
    def forward(self, hidden_states):
        pass

    def set_layer_number(self, layer_number: int):
        self.layer_number = layer_number
        self.router.set_layer_number(layer_number)


class HeterMoELayer(BaseMoELayer):
    """Mixture of experts Layer **currently only supports no token dropping**.

    Args:
        BaseMoELayer (MegatronModule): Base class for MoE layers
    """

    def __init__(
        self, config: TransformerConfig, submodules: MLPSubmodules = None, layer_number: int = None
    ):
        self.submodules = submodules
        super(HeterMoELayer, self).__init__(config=config, layer_number=layer_number)
        self.router = TopKRouter(config=self.config)

        ## token_dispather type
        if config.moe_token_dispatcher_type == "allgather":
            self.token_dispatcher = MoEAllGatherTokenDispatcher(
                self.num_local_experts, self.local_expert_indices, config=self.config
            )
        elif config.moe_token_dispatcher_type == "alltoall":
            self.token_dispatcher = MoEAlltoAllTokenDispatcher(
                self.num_local_experts, self.local_expert_indices, config=self.config
            )
        else:
            raise ValueError(
                f"Unsupported token dispatcher type: {config.moe_token_dispatcher_type}"
            )
        ## mindspeed-llm部分
        global_args = get_args()

        if global_args.moe_intermediate_size:
            self.config.ffn_hidden_size = global_args.moe_intermediate_size
        
        if self.config.moe_grouped_gemm:
            self.experts = GroupedMLP(self.num_local_experts, self.config)
        else:
            assert isinstance(self.submodules, MLPSubmodules)
            self.experts = SequentialMLP(self.num_local_experts, self.config, self.submodules)

        if global_args.n_shared_experts:
            config = deepcopy(self.config)
            config.ffn_hidden_size = global_args.n_shared_experts * self.config.ffn_hidden_size
            self.shared_experts = MLP(config, MLPSubmodules(linear_fc1=ColumnParallelLinear,
                                                                 linear_fc2=RowParallelLinear,))
            # For using layer_number when recompute activation function is enabled.
            self.shared_experts.layer_number = self.layer_number
            if global_args.shared_expert_gate:
                self.shared_expert_gate = build_module(
                    RowParallelLinear,
                    config.hidden_size,
                    global_args.shared_expert_gate_output_dimension,
                    config=config,
                    init_method=config.output_layer_init_method,
                    bias=None,
                    input_is_parallel=True,
                    skip_bias_add=True
                )


    def forward(self):
        # process MoE

        ## Attention->Expert Communication

        hidden_states = None

        scores, indices = self.router(hidden_states)
        ## Expert<->Expert Communication
        (dispatched_input, tokens_per_expert) = self.token_dispatcher.token_permutation(
        hidden_states, scores, indices
        )
        ## Expert compute
        router_expert_output, mlp_bias = self.experts(dispatched_input, tokens_per_expert)
        ## Expert<->Expert Communication
        output, mlp_bias = self.token_dispatcher.token_unpermutation(router_expert_output, mlp_bias)
    
        ## Expert->Attention Communication



        args = get_args()

        ## 负载均衡损失
        if args.moe_router_load_balancing_type == "group_limited_greedy":
            save_to_aux_losses_tracker(
                "load_balancing_loss",
                self.router.l_aux,
                self.layer_number,
                self.config.num_layers,
            )
            save_to_aux_losses_tracker(
                "load_balancing_expert_level_loss",
                    self.router.l_expert_aux / args.moe_aux_loss_coeff,
                self.layer_number,
                self.config.num_layers,
            )
            if hasattr(self.router, 'l_device_aux'):
                save_to_aux_losses_tracker(
                    "load_balancing_device_level_loss",
                    self.router.l_device_aux / args.moe_device_level_aux_loss_coeff,
                    self.layer_number,
                    self.config.num_layers,
                )
            if hasattr(self.router, 'l_comm_aux'):
                save_to_aux_losses_tracker(
                    "load_balancing_comm_level_loss",
                    self.router.l_comm_aux / args.moe_comm_aux_loss_coeff,
                    self.layer_number,
                    self.config.num_layers,
                )
            output = MoEAuxLossAutoScaler.apply(output, self.router.l_aux)
    
        if args.n_shared_experts:
            share_experts_output, share_experts_bias = self.shared_experts(hidden_states)
            if args.shared_expert_gate:
                share_experts_output = F.sigmoid(self.shared_expert_gate(hidden_states)[0]) * share_experts_output
            output = output + share_experts_output
        
            if self.token_dispatcher.add_bias:
                mlp_bias = mlp_bias + share_experts_bias

        return None
    




