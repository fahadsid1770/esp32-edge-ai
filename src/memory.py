import gc
import threading

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint


class CPUOffload:
    def __init__(self, model, device, offload_optimizer=True):
        self.model = model
        self.device = device
        self.offload_optimizer = offload_optimizer
        self.hooks = []
        self.param_to_cpu = {}
        self.param_to_gpu = {}
        self._setup_hooks()

    def _setup_hooks(self):
        for p in self.model.parameters():
            if p.requires_grad:
                p.register_post_accumulate_grad_hook(self._make_hook(p))

    def _make_hook(self, param):
        def hook(grad):
            if param.device != torch.device("cpu"):
                self.param_to_cpu[param] = param.data.to("cpu")
                if param.grad is not None:
                    param._cpu_grad = param.grad.to("cpu")
                param.data = self.param_to_cpu[param]
                param.grad = None
                torch.cuda.empty_cache()

        return hook

    def offload_to_cpu(self):
        for p in self.model.parameters():
            if p.requires_grad and p.device != torch.device("cpu"):
                self.param_to_cpu[p] = p.data.to("cpu")
                p.data = self.param_to_cpu[p]
        torch.cuda.empty_cache()

    def restore_to_gpu(self):
        for p in self.model.parameters():
            if p.requires_grad and p in self.param_to_cpu:
                p.data = self.param_to_cpu[p].to(self.device)
                if hasattr(p, "_cpu_grad"):
                    p.grad = p._cpu_grad.to(self.device)
                    delattr(p, "_cpu_grad")
        torch.cuda.empty_cache()

    def cleanup(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()


class DynamicMemoryManager:
    def __init__(self, model, optimizer=None, device=None, memory_threshold=0.8):
        self.model = model
        self.optimizer = optimizer
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.memory_threshold = memory_threshold
        self.cpu_offload = None
        self.use_cpu_offload = False
        self.checkpoint_blocks = False
        self._orig_forward = None

    def get_gpu_memory_usage(self):
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            return allocated, reserved, total
        return 0.0, 0.0, 0.0

    def should_offload(self):
        if not torch.cuda.is_available():
            return False
        allocated, _, total = self.get_gpu_memory_usage()
        return (allocated / total) > self.memory_threshold

    def enable_cpu_offload(self):
        if not self.use_cpu_offload:
            self.cpu_offload = CPUOffload(self.model, self.device)
            self.use_cpu_offload = True
            print("[DynamicMemory] CPU offload enabled")

    def disable_cpu_offload(self):
        if self.use_cpu_offload and self.cpu_offload:
            self.cpu_offload.restore_to_gpu()
            self.cpu_offload.cleanup()
            self.cpu_offload = None
            self.use_cpu_offload = False
            print("[DynamicMemory] CPU offload disabled")

    def enable_gradient_checkpointing(self):
        self.checkpoint_blocks = True
        self._apply_gradient_checkpointing()
        print("[DynamicMemory] Gradient checkpointing enabled")

    def _apply_gradient_checkpointing(self):
        if hasattr(self.model, "blocks"):
            for block in self.model.blocks:
                block.forward = self._wrap_block_forward(block)

    def _wrap_block_forward(self, block):
        def checkpointed_forward(x, cos, sin, ple=None, use_checkpoint=None):
            return checkpoint(
                block._original_forward,
                x, cos, sin, ple,
                use_reentrant=False,
                preserve_rng_state=True
            )
        block._original_forward = block.forward
        return checkpointed_forward

    def step(self, force_offload=False):
        if (force_offload or self.should_offload()) and not self.use_cpu_offload:
            self.enable_cpu_offload()

        if self.use_cpu_offload:
            if self.cpu_offload and not self.should_offload():
                self.disable_cpu_offload()
            gc.collect()
            torch.cuda.empty_cache()

    def print_memory_stats(self, step=0):
        if torch.cuda.is_available():
            allocated, reserved, total = self.get_gpu_memory_usage()
            print(
                f"[Memory] Step {step} | GPU: {allocated:.2f}GB / {total:.2f}GB "
                f"(reserved: {reserved:.2f}GB) | CPU offload: {self.use_cpu_offload}"
            )

    def cleanup(self):
        if self.use_cpu_offload:
            self.disable_cpu_offload()


def apply_gradient_checkpointing(model):
    if hasattr(model, "blocks"):
        for block in model.blocks:
            if hasattr(block, "forward") and not hasattr(block, "_original_forward"):
                original_forward = block.forward
                def make_checkpointed(ofwd):
                    def checkpointed(x, cos, sin, ple=None, use_checkpoint=None):
                        return checkpoint(
                            ofwd, x, cos, sin, ple,
                            use_reentrant=False,
                            preserve_rng_state=True
                        )
                    return checkpointed
                block._original_forward = original_forward
                block.forward = make_checkpointed(original_forward)
    return model
