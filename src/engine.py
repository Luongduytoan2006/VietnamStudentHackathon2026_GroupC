# -*- coding: utf-8 -*-
"""Wrapper quanh llama_cpp.Llama: load model, sinh văn bản (±grammar 1 token)."""
from .bootstrap import preload

preload()  # phải gọi TRƯỚC khi import llama_cpp (Windows)
from llama_cpp import Llama, LlamaGrammar  # noqa: E402

from .io_utils import LETTERS  # noqa: E402


class Engine:
    def __init__(self, model_path, n_gpu_layers=-1, n_ctx=4096, flash_attn=True,
                 verbose=False, n_threads=None):
        self.model_path = model_path
        self.llm = Llama(
            model_path=model_path,
            n_gpu_layers=n_gpu_layers,
            n_ctx=n_ctx,
            flash_attn=flash_attn,
            verbose=verbose,
            n_threads=n_threads,
        )
        # cache grammar [A-last] theo số lựa chọn
        self._gcache = {}

    # ---- grammar 1 token ----
    def grammar_for(self, n):
        if n not in self._gcache:
            last = LETTERS[n - 1]
            self._gcache[n] = LlamaGrammar.from_string(f"root ::= [A-{last}]")
        return self._gcache[n]

    # ---- sinh có chat template ----
    def chat(self, messages, max_tokens=8, temperature=0.0, top_p=1.0,
             grammar=None, stop=None, seed=None):
        kw = dict(messages=messages, max_tokens=max_tokens,
                  temperature=temperature, top_p=top_p)
        if grammar is not None:
            kw["grammar"] = grammar
        if stop is not None:
            kw["stop"] = stop
        if seed is not None:
            kw["seed"] = seed
        out = self.llm.create_chat_completion(**kw)
        return out["choices"][0]["message"]["content"]
