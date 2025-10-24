# rank_manager.py
import torch
from megatron.training import get_args, print_rank_0
# def _add_herterMoE_args(parser):
#     group = parser.add_argument_group(title='herterMoE')

#     group.add_argument('--herter-moe-enable', action='store_true', default=False,type=bool,
#                        help='Enable heteroMoE layers in the model.')
#     group.add_argument('--herter-moe-attention-world-size', type=int, default=1,
#                        help='The attention rank world size for heteroMoE layers.')
#     group.add_argument('--herter-moe-ffn-world-size', type=int, default=1,
#                        help='The ffn rank world size for heteroMoE layers.')
#     return parser
_initialized = False
_attention_world_size = None
_ffn_world_size = None
_world_size = None
_rank = None

def set_heteroMoE_config(args):
    global _initialized, _world_size, _rank, _ffn_world_size, _attention_world_size
    if not args.herter_moe_enable:
        return
    if not _initialized:
        _world_size = torch.distributed.get_world_size()
        _rank = torch.distributed.get_rank()
        _ffn_world_size = args.herter_moe_ffn_world_size
        _attention_world_size = args.herter_moe_attention_world_size
        assert _attention_world_size + _ffn_world_size == _world_size, \
            f"herterMoE world size mismatch: attention_world_size ({_attention_world_size}) + ffn_world_size ({_ffn_world_size}) != world_size ({_world_size})"
        # print_rank_0(f"herterMoE rank config: attention_world_size={_attention_world_size}, ffn_world_size={_ffn_world_size}, total_world_size={_world_size}")
        if _rank < _attention_world_size:
            args.herter_moe_is_A= True
            args.herter_moe_is_E= False
        else:   
            args.herter_moe_is_E= True
            args.herter_moe_is_A= False
    _initialized = True


