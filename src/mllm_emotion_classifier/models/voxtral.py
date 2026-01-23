"""Voxtral wrapper for Emotion Recognition."""

import os
import logging
import torch

from typing import List, Union
from .base import BaseEmotionModel
from transformers import VoxtralForConditionalGeneration, VoxtralProcessor, set_seed
from .prompts.chat_voxtral import build_conversation
from .postprocess import postprocess_ser_response

logger = logging.getLogger(__name__)

DEFAULT_EMOTIONS = ["Happy", "Sad", "Angry", "Neutral", "Fear", "Disgust", "Surprise"]


class VoxtralEmotionWrapper(BaseEmotionModel):
    
    def __init__(
        self,
        trust_remote_code: bool = True,
        torch_dtype: str = 'bfloat16',
        max_new_tokens: int = 5,
        min_new_tokens: int = 1,
        temperature: float = 1.0,
        top_p: float = 1.0,
        do_sample: bool = True,
        class_labels = None,
        prompt_name: str = "user_labels",
        device: str = "cuda",
        seed: int = 42,
        **kwargs,
    ):
        super().__init__()
        set_seed(seed)
        self.seed = seed

        self.name = "voxtral"
        self.checkpoint = "mistralai/Voxtral-Mini-3B-2507"
        self.trust_remote_code = trust_remote_code
        self.torch_dtype = getattr(torch, torch_dtype) if isinstance(torch_dtype, str) else torch_dtype
        self.max_new_tokens = max_new_tokens
        self.min_new_tokens = min_new_tokens
        self.do_sample = do_sample
        self.device = device
        self.temperature = temperature
        self.top_p = top_p
        # os.environ['TOKENIZERS_PARALLELISM'] = 'false'
        self.processor = VoxtralProcessor.from_pretrained(
            self.checkpoint,
            # trust_remote_code=True,
            # use_fast=False 
        )
        self.model = VoxtralForConditionalGeneration.from_pretrained(
            self.checkpoint,
            device_map=self.device,
            dtype=self.torch_dtype
        )
        self.model.eval()

        self.class_labels = list(class_labels) if class_labels is not None else DEFAULT_EMOTIONS
        self.letter_to_label = {label[0].upper(): label for label in self.class_labels}

        self.prompt_name = prompt_name

    def collate_fn(self, inputs):
        input_audios = [_['audio'] for _ in inputs]
        labels = [_['label'] for _ in inputs]

        labels_str = ", ".join(self.class_labels)

        conversations = []
        for audio_path in input_audios:
            conversation = build_conversation(self.prompt_name, labels_str, audio_path)
            conversations.append(conversation)

        processed_inputs = self.processor.apply_chat_template(conversations)
        
        return processed_inputs, labels
    
    def _decode_outputs(
        self,
        input_ids: torch.Tensor,
        output_ids: torch.Tensor,
    ) -> List[str]:
        output_ids = output_ids[:, input_ids.shape[1]:]
        outputs = self.processor.batch_decode(
            output_ids, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=False
        )
        return outputs

    def predict(self, inputs: dict) -> List[Union[str, None]]:
        set_seed(self.seed)
        
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                min_new_tokens=self.min_new_tokens,
                do_sample=self.do_sample,
                temperature=self.temperature,
                top_p=self.top_p,
            )
        outputs = self._decode_outputs(inputs['input_ids'], output_ids)
        predictions = postprocess_ser_response(
            class_labels=self.class_labels,
            model_responses=outputs,
        )
        return predictions

    def forward(self, inputs: dict):
        return self.model(**inputs)