import time

from typing import Dict, Optional, List, Callable, Union
from types import MethodType

import torch
import torch.nn.functional as F

from transformers.generation.utils import (
    GenerateBeamOutput, GenerateBeamEncoderDecoderOutput, GenerateBeamDecoderOnlyOutput,
    _split_model_inputs, stack_model_outputs
)
from transformers.generation.logits_process import LogitsProcessorList, LogitsProcessor
from transformers.generation.stopping_criteria import StoppingCriteriaList
from transformers.generation.beam_search import BeamScorer
from transformers.generation.configuration_utils import GenerationConfig
from transformers.cache_utils import Cache
from transformers.modeling_outputs import CausalLMOutputWithPast



def replace_beam_search_with_TreeAttn(model):
    model._beam_search = MethodType(_beam_search, model)
    return model

def _beam_search(
    self,
    input_ids: torch.LongTensor,
    beam_scorer: BeamScorer,
    logits_processor: LogitsProcessorList,
    stopping_criteria: StoppingCriteriaList,
    generation_config: GenerationConfig,
    synced_gpus: bool,
    logits_warper: Optional[LogitsProcessorList] = None,
    **model_kwargs,
) -> Union[GenerateBeamOutput, torch.LongTensor]:

    # init values
    pad_token_id = generation_config.pad_token_id
    eos_token_id = generation_config.eos_token_id
    output_attentions = generation_config.output_attentions
    output_hidden_states = generation_config.output_hidden_states
    output_scores = generation_config.output_scores
    output_logits = generation_config.output_logits
    return_dict_in_generate = generation_config.return_dict_in_generate
    sequential = generation_config.low_memory
    do_sample = generation_config.do_sample
    if do_sample is True and not isinstance(logits_warper, LogitsProcessorList):
        raise ValueError(
            "`do_sample` is set to `True`, `logits_warper` must be a `LogitsProcessorList` instance (it is "
            f"{logits_warper})."
        )

    batch_size = len(beam_scorer._beam_hyps)
    assert batch_size == 1, "Beam search by speculative decoding only supports batch_size = 1 currently!"
    num_beams = beam_scorer.num_beams

    batch_beam_size, cur_len = input_ids.shape

    input_ids = input_ids[:1]  # cancel expanded input_ids in transformers
    model_kwargs = self._get_initial_cache_position(input_ids, model_kwargs)

    if num_beams * batch_size != batch_beam_size:
        raise ValueError(
            f"Batch dimension of `input_ids` should be {num_beams * batch_size}, but is {batch_beam_size}."
        )

    # init attention / hidden states / scores tuples
    scores = () if (return_dict_in_generate and output_scores) else None
    raw_logits = () if (return_dict_in_generate and output_logits) else None
    beam_indices = (
        tuple(() for _ in range(batch_beam_size)) if (return_dict_in_generate and output_scores) else None
    )
    decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
    cross_attentions = () if (return_dict_in_generate and output_attentions) else None
    decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

    # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
    if return_dict_in_generate and self.config.is_encoder_decoder:
        encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
        encoder_hidden_states = (
            model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
        )

    # initialise
    beam_size = num_beams
    min_dtype = -1e9
    device = input_ids.device
    causal_mask = (torch.tril(torch.ones((cur_len, cur_len), device=device)) == 0) * min_dtype
    model_inputs = {
        "input_ids": input_ids,
        "attention_mask": causal_mask[None, None, :, :],
        "position_ids": torch.arange(cur_len, device=device).unsqueeze(0),
        "past_key_values": None
    }
    beam_scores = torch.zeros(1, dtype=torch.float, device=device)
    beam_sequences = input_ids.repeat(beam_size, 1)

    this_peer_finished = False

    first = True
    while self._has_unfinished_sequences(this_peer_finished, synced_gpus, device=input_ids.device):
        if synced_gpus and this_peer_finished:
            cur_len = cur_len + 1
            continue  # don't waste resources running the code we don't need

        # if sequential is True, split the input to batches of batch_size and run sequentially
        if sequential:
            raise RuntimeError(
                f"Currently low_memory beam search by speculative decoding is not supported "
            )
        else:  # Unchanged original behavior
            outputs = self(
                **model_inputs,
                return_dict=True,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )

        n_last_beam = len(beam_scores)
        next_token_logits = outputs.logits[0, -n_last_beam:]  # [batch_size * beam_size, vocab_size]
        
        next_token_scores = F.log_softmax(next_token_logits, dim=-1)
        if first:
            next_token_scores = logits_processor(beam_sequences, next_token_scores.repeat(beam_size, 1))[:1]
            first = False
        else:
            next_token_scores = logits_processor(beam_sequences, next_token_scores)
        if do_sample:
            next_token_scores = logits_warper(beam_sequences, next_token_scores)
        
        # Store scores, attentions and hidden_states when required
        if return_dict_in_generate:
            if output_scores:
                scores += (next_token_scores,)
            if output_logits:
                raw_logits += (next_token_logits,)
            if output_attentions:
                decoder_attentions += (
                    (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                )
                if self.config.is_encoder_decoder:
                    cross_attentions += (outputs.cross_attentions,)
            if output_hidden_states:
                decoder_hidden_states += (
                    (outputs.decoder_hidden_states,)
                    if self.config.is_encoder_decoder
                    else (outputs.hidden_states,)
                )
        vocab_size = next_token_scores.shape[-1]
        beam_scores = next_token_scores + beam_scores[:, None].expand_as(next_token_scores)
        beam_scores = beam_scores.view(-1)
        if do_sample:
            probs = F.softmax(beam_scores, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=beam_size*2)  # Sample beam_size * 2 tokens as transformers does
            beam_scores = torch.gather(beam_scores, -1, next_tokens)
            beam_scores, _indices = torch.sort(beam_scores, descending=True, dim=-1)
            next_tokens = torch.gather(next_tokens, -1, _indices)
            next_tokens = next_tokens[:beam_size]
            beam_scores = beam_scores[:beam_size]
        else:
            beam_scores, next_tokens = beam_scores.topk(beam_size)
        beam_indices = next_tokens // vocab_size
        beam_tokens = next_tokens % vocab_size

        beam_sequences = torch.cat((beam_sequences[beam_indices], beam_tokens[:, None]), dim=-1)
        causal_mask = torch.cat((model_inputs["attention_mask"][0, 0, -n_last_beam:][beam_indices], (torch.eye(beam_size, device=device) == 0) * min_dtype), dim=-1)
        causal_mask = causal_mask[None, None, :, :]
        position_ids = (model_inputs["position_ids"][:, -1:] + 1).repeat(1, beam_size)

        cur_len = cur_len + 1

        # TODO: Currently not consider eos in beam_sequencess.
        if all(stopping_criteria(beam_sequences, scores)):
            this_peer_finished = True

        model_inputs = {
            "input_ids": beam_tokens[None, :],
            "attention_mask": causal_mask,
            "position_ids": position_ids,
            "past_key_values": outputs.past_key_values,
        }
    
    beam_sequences = beam_sequences[:self.generation_config.num_return_sequences]

    if return_dict_in_generate:
        if not output_scores:
            beam_scores = None

        if self.config.is_encoder_decoder:
            return GenerateBeamEncoderDecoderOutput(
                sequences=beam_sequences,
                sequences_scores=beam_scores,
                scores=scores,
                logits=raw_logits,
                beam_indices=beam_indices,
                encoder_attentions=encoder_attentions,
                encoder_hidden_states=encoder_hidden_states,
                decoder_attentions=decoder_attentions,
                cross_attentions=cross_attentions,
                decoder_hidden_states=decoder_hidden_states,
                past_key_values=outputs.past_key_values,
            )
        else:
            return GenerateBeamDecoderOnlyOutput(
                sequences=beam_sequences,
                sequences_scores=beam_scores,
                scores=scores,
                logits=raw_logits,
                beam_indices=beam_indices,
                attentions=decoder_attentions,
                hidden_states=decoder_hidden_states,
                past_key_values=outputs.past_key_values,
            )
    else:
        return beam_sequences


