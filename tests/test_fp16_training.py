"""Exercise FP16 native AMP with local, randomly initialized tiny BERT weights."""
from types import SimpleNamespace

import pytest
import torch
from transformers import BertConfig, BertModel

from tevatron.retriever.arguments import ModelArguments, TevatronTrainingArguments
from tevatron.retriever.driver.train import _get_model_dtype
from tevatron.retriever.modeling import DenseModel


@pytest.mark.parametrize('fp16,bf16,lora,adapter,expected', [
    (False, False, False, None, torch.float32),
    (True, False, False, None, torch.float32),
    (False, True, False, None, torch.bfloat16),
    (True, False, True, None, torch.float16),
    (True, False, False, 'local-adapter', torch.float16),
])
def test_model_dtype(fp16, bf16, lora, adapter, expected):
    model_args = ModelArguments('local-model', lora=lora, lora_name_or_path=adapter)
    training_args = SimpleNamespace(fp16=fp16, bf16=bf16)
    assert _get_model_dtype(model_args, training_args) == expected


@pytest.fixture
def tiny_checkpoint(tmp_path):
    with torch.random.fork_rng():
        torch.manual_seed(23)
        model = BertModel(BertConfig(
            vocab_size=32, hidden_size=16, num_hidden_layers=1,
            num_attention_heads=2, intermediate_size=24,
            hidden_dropout_prob=0, attention_probs_dropout_prob=0,
        ))
        path = tmp_path / 'tiny-bert'
        model.save_pretrained(path)
    return path


def amp_step(model):
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    scaler = torch.amp.GradScaler('cpu', init_scale=128)
    query = {'input_ids': torch.tensor([[1, 2, 3], [4, 5, 6]]),
             'attention_mask': torch.ones(2, 3, dtype=torch.long)}
    passage = {'input_ids': torch.tensor([[1, 2, 3], [7, 8, 9], [4, 5, 6], [10, 11, 12]]),
               'attention_mask': torch.ones(4, 3, dtype=torch.long)}
    before = {name: p.detach().clone() for name, p in model.named_parameters() if p.requires_grad}
    with torch.autocast('cpu', dtype=torch.float16):
        loss = model(query=query, passage=passage).loss
    assert torch.isfinite(loss)
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    assert gradients and all(torch.isfinite(g).all() for g in gradients)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
    scaler.step(optimizer)
    scaler.update()
    assert any(not torch.equal(before[name], p) for name, p in model.named_parameters() if p.requires_grad)


def build_model(checkpoint, output_dir, *, dtype=None, lora=False, adapter=None):
    model_args = ModelArguments(str(checkpoint), lora=lora,
                                lora_name_or_path=adapter, lora_target_modules='query,value')
    train_args = TevatronTrainingArguments(output_dir=str(output_dir), use_cpu=True,
                                           report_to='none', fp16=True)
    dtype = _get_model_dtype(model_args, train_args) if dtype is None else dtype
    return DenseModel.build(model_args, train_args, torch_dtype=dtype,
                            attn_implementation='sdpa', local_files_only=True)


def test_fp16_full_finetuning_amp_step(tiny_checkpoint, tmp_path):
    # Actual CPU GradScaler exercises the same dtype guard as native CUDA AMP.
    # CPU Trainer does not enable a scaler, so this is deliberately not a Trainer test.
    model = build_model(tiny_checkpoint, tmp_path / 'training')
    assert all(p.dtype == torch.float32 for p in model.parameters() if p.requires_grad)
    amp_step(model)


def test_old_fp16_weights_reproduce_unscale_error(tiny_checkpoint, tmp_path):
    model = build_model(tiny_checkpoint, tmp_path / 'training', dtype=torch.float16)
    with pytest.raises(ValueError, match='Attempting to unscale FP16 gradients'):
        amp_step(model)


def test_fp16_lora_and_loaded_adapter(tiny_checkpoint, tmp_path):
    model = build_model(tiny_checkpoint, tmp_path / 'training', lora=True)
    assert all(p.dtype == torch.float32 for p in model.parameters() if p.requires_grad)
    assert any(p.dtype == torch.float16 for p in model.parameters() if not p.requires_grad)
    amp_step(model)
    adapter = tmp_path / 'adapter'
    model.encoder.save_pretrained(adapter)
    loaded = build_model(tiny_checkpoint, tmp_path / 'reloaded-training', adapter=str(adapter))
    assert all(p.dtype == torch.float32 for p in loaded.parameters() if p.requires_grad)
    assert any(p.dtype == torch.float16 for p in loaded.parameters() if not p.requires_grad)
    amp_step(loaded)
