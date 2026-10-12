"""Preserve native clipping, with FP64 norm fallback for finite huge gradients.

This does not sanitize nonfinite gradients. Those remain a hard failure. Only
overflow of an FP32 reduction over otherwise finite gradient values is fixed.
"""
import hashlib
from pathlib import Path
import torch

ORIGINAL = torch.nn.utils.clip_grad_norm_
EVENTS = []
CALLS = 0

def install(callback=None):
    metadata={'name':'finite-gradient-fp64-norm-fallback-v1',
        'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    if getattr(torch.nn.utils.clip_grad_norm_,'_whale_safe_norm',False):return metadata
    @torch.no_grad()
    def clip(parameters,max_norm,norm_type=2.0,error_if_nonfinite=False,foreach=None):
        global CALLS
        params=[parameters] if torch.is_tensor(parameters) else list(parameters)
        grads=[p.grad for p in params if p.grad is not None]
        index=CALLS;CALLS+=1
        if not grads or float(norm_type)!=2.0:
            return ORIGINAL(params,max_norm,norm_type,error_if_nonfinite,foreach)
        norms=torch._foreach_norm(grads,2)
        total=torch.linalg.vector_norm(torch.stack(norms),2)
        if torch.isfinite(total):
            return ORIGINAL(params,max_norm,norm_type,error_if_nonfinite,foreach)
        bad=[i for i,g in enumerate(grads) if not torch.isfinite(g).all()]
        event={'minibatch_index':index,'norm_fp32':str(float(total)),
            'nonfinite_gradient_tensor_indices':bad}
        if bad:
            event['action']='fail_without_optimizer_step'
            EVENTS.append(event)
            if callback:callback(event)
            raise FloatingPointError('Nonfinite gradient entries, not just norm overflow')
        precise=torch.linalg.vector_norm(torch.stack([
            torch.linalg.vector_norm(g,2,dtype=torch.float64) for g in grads]),2)
        assert torch.isfinite(precise)
        coefficient=(float(max_norm)/(precise+1e-6)).clamp(max=1.0)
        for g in grads:g.mul_(coefficient.to(device=g.device,dtype=g.dtype))
        event.update(action='clip_finite_gradients_using_fp64_norm',norm_fp64=float(precise),
            clip_coefficient=float(coefficient))
        EVENTS.append(event)
        if callback:callback(event)
        return precise
    clip._whale_safe_norm=True
    torch.nn.utils.clip_grad_norm_=clip
    return metadata
