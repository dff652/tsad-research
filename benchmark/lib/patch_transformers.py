
import sys

def apply_loss_kwargs_patch():
    # 尝试导入 transformers.utils 以检查 LossKwargs
    try:
        import transformers.utils
        if not hasattr(transformers.utils, "LossKwargs"):
            from typing import TypedDict
            
            # 定义为 TypedDict 以避免 TypeError: cannot inherit from both a TypedDict type and a non-TypedDict base class
            class LossKwargs(TypedDict):
                loss: float
                
            transformers.utils.LossKwargs = LossKwargs
            sys.modules["transformers.utils.LossKwargs"] = LossKwargs
            print("[Patch] Applied transformers.utils.LossKwargs patch (TypedDict)")
    except ImportError:
        pass
    except Exception as e:
        print(f"[Patch] Failed to apply LossKwargs patch: {e}")

def apply_dynamic_cache_patch():
    """
    Fix for AttributeError: 'DynamicCache' object has no attribute 'seen_tokens'
    Caused by transformers >= 4.45 compatibility issues with older Qwen models
    """
    try:
        from transformers.cache_utils import DynamicCache
        if not hasattr(DynamicCache, "seen_tokens"):
            # Map seen_tokens to get_seq_length()
            # Note: get_seq_length() returns the number of tokens currently in the cache
            print("[Patch] DynamicCache.seen_tokens is missing, patching it to use get_seq_length()")
            
            @property
            def seen_tokens(self):
                return self.get_seq_length()
                
            DynamicCache.seen_tokens = seen_tokens
            print("[Patch] Applied transformers.cache_utils.DynamicCache.seen_tokens patch")
    except ImportError:
        pass
    except Exception as e:
        print(f"[Patch] Failed to apply DynamicCache patch: {e}")

# 在模块加载时自动应用
apply_loss_kwargs_patch()
apply_dynamic_cache_patch()

# 诊断信息：检查环境和模块路径
try:
    import sys
    import os
    print(f"[Patch] sys.path: {sys.path}", flush=True)
    print(f"[Patch] patch_transformers path: {os.path.abspath(__file__)}", flush=True)
    
    import chatts_detect
    print(f"[Patch] chatts_detect loaded from: {chatts_detect.__file__}", flush=True)
except Exception as e:
    print(f"[Patch] Diagnostic failed: {e}", flush=True)
