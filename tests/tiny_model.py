"""Build a tiny, randomly initialised Qwen3-VL model + tokenizer + image processor on disk.

Used by the tests and the CI so the whole pipeline (train, calibrate, decide, serve, Decision Index
engine) runs on a CPU in seconds without downloading anything. It has the same architecture and
code path as Qwen/Qwen3-VL-2B-Instruct, just a few thousand times smaller.
"""
from __future__ import annotations

import os

SPECIAL = ["<|endoftext|>", "<|vision_start|>", "<|image_pad|>", "<|vision_end|>", "<|pad|>"]


def build_tiny(out_dir: str, seed: int = 0) -> str:
    import torch
    from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers
    from transformers import PreTrainedTokenizerFast, Qwen3VLConfig, Qwen3VLModel
    from transformers.models.qwen2_vl.image_processing_pil_qwen2_vl import Qwen2VLImageProcessorPil

    if os.path.exists(os.path.join(out_dir, "config.json")):
        return out_dir
    os.makedirs(out_dir, exist_ok=True)
    torch.manual_seed(seed)

    corpus = [
        "State: Question: Options: Answer: yes no choice score the a an is of to in and for with",
        "billing shipping technical general urgent refund customer package damaged arrived month",
        "Which category has the highest value? Is there a red square? How readable is this chart?",
        "0 1 2 3 4 5 6 7 8 9 A B C D E positive negative neutral World Sports Business Science",
        "abcdefghijklmnopqrstuvwxyz ABCDEFGHIJKLMNOPQRSTUVWXYZ .,:;!?-_()[]{}\"' \n",
    ] * 20
    tk = Tokenizer(models.BPE(unk_token=None))
    tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tk.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=600, special_tokens=SPECIAL,
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    tk.train_from_iterator(corpus, trainer)
    tok = PreTrainedTokenizerFast(tokenizer_object=tk, eos_token="<|endoftext|>", pad_token="<|pad|>",
                                  additional_special_tokens=SPECIAL[1:4])
    tok.save_pretrained(out_dir)

    ip = Qwen2VLImageProcessorPil(patch_size=16, merge_size=2, temporal_patch_size=2,
                                  min_pixels=32 * 32, max_pixels=256 * 32 * 32)
    ip.save_pretrained(out_dir)

    ids = {t: tok.convert_tokens_to_ids(t) for t in SPECIAL}
    cfg = Qwen3VLConfig(
        text_config=dict(vocab_size=len(tok), hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                         num_attention_heads=4, num_key_value_heads=2, head_dim=16,
                         max_position_embeddings=4096,
                         rope_parameters={"rope_type": "default", "rope_theta": 10000.0,
                                          "mrope_section": [4, 2, 2], "mrope_interleaved": True}),
        vision_config=dict(depth=2, hidden_size=32, intermediate_size=64, num_heads=2, patch_size=16,
                           spatial_merge_size=2, temporal_patch_size=2, out_hidden_size=64,
                           num_position_embeddings=256, deepstack_visual_indexes=[0, 1]),
        image_token_id=ids["<|image_pad|>"], vision_start_token_id=ids["<|vision_start|>"],
        vision_end_token_id=ids["<|vision_end|>"], video_token_id=ids["<|pad|>"],
    )
    Qwen3VLModel(cfg).save_pretrained(out_dir)
    return out_dir


if __name__ == "__main__":
    import sys
    print(build_tiny(sys.argv[1] if len(sys.argv) > 1 else "runs/tiny-base"))
