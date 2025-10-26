# rank_manager.py
import torch
from megatron.training import get_args, print_rank_0
# def _add_heterMoE_args(parser):
#     group = parser.add_argument_group(title='heterMoE')

#     group.add_argument('--heter-moe-enable', action='store_true', default=False,type=bool,
#                        help='Enable heteroMoE layers in the model.')
#     group.add_argument('--heter-moe-attention-world-size', type=int, default=1,
#                        help='The attention rank world size for heteroMoE layers.')
#     group.add_argument('--heter-moe-ffn-world-size', type=int, default=1,
#                        help='The ffn rank world size for heteroMoE layers.')
#     return parser
_initialized = False
_attention_world_size = None
_ffn_world_size = None
_world_size = None
_rank = None

def set_heteroMoE_config(args):
    global _initialized, _world_size, _rank, _ffn_world_size, _attention_world_size
    if not args.heter_moe_enable:
        return
    if not _initialized:
        _world_size = torch.distributed.get_world_size()
        _rank = torch.distributed.get_rank()
        _ffn_world_size = args.heter_moe_ffn_world_size
        _attention_world_size = args.heter_moe_attention_world_size
        assert _attention_world_size + _ffn_world_size == _world_size, \
            f"heterMoE world size mismatch: attention_world_size ({_attention_world_size}) + ffn_world_size ({_ffn_world_size}) != world_size ({_world_size})"
        # print_rank_0(f"heterMoE rank config: attention_world_size={_attention_world_size}, ffn_world_size={_ffn_world_size}, total_world_size={_world_size}")
        if _rank < _attention_world_size:
            args.heter_moe_is_A= True
            args.heter_moe_is_E= False
        else:   
            args.heter_moe_is_E= True
            args.heter_moe_is_A= False
    _initialized = True
# 用于heterMoE debug
def heterMoE_debug_print(msg, rank=0):
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == rank:
            print(msg, flush=True)
    else:
        print(msg, flush=True)
