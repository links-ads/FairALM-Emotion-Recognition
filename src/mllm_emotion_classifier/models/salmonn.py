"""SALMONN wrapper for Emotion Recognition."""

import logging
import torch
from typing import List, Union
from .base import BaseEmotionModel
from transformers import set_seed
from .prompts.chat_qwen import build_conversation
from .postprocess import postprocess_ser_response
from SALMONN_7B.model import SALMONN

logger = logging.getLogger(__name__)

DEFAULT_EMOTIONS = ["Happy", "Sad", "Angry", "Neutral", "Fear", "Disgust", "Surprise"]


class SALMONNEmotionWrapper(BaseEmotionModel):
    
    def __init__(
        self,
        ckpt: str = None,
        whisper_path: str = None,
        beats_path: str = None,
        vicuna_path: str = None,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_p: float = 1.0,
        do_sample: bool = True,
        num_beams: int = 4,
        class_labels=None,
        prompt_name: str = "user_labels",
        device: str = "cuda",
        seed: int = 42,
        lora_alpha: int = 32,
        low_resource: bool = False,
        **kwargs,
    ):
        super().__init__()
        set_seed(seed)
        self.seed = seed

        self.name = "salmonn"
        self.max_new_tokens = max_new_tokens
        self.do_sample = do_sample
        self.temperature = temperature
        self.top_p = top_p
        self.num_beams = num_beams
        self.device = device

        self.model = SALMONN(
            ckpt=ckpt,
            whisper_path=whisper_path,
            beats_path=beats_path,
            vicuna_path=vicuna_path,
            lora_alpha=lora_alpha,
            low_resource=low_resource
        )
        self.model.to(self.device)
        self.model.eval()

        self.class_labels = list(class_labels) if class_labels is not None else DEFAULT_EMOTIONS
        self.prompt_name = prompt_name

    def _build_prompt(self, emotions_list: List[str]) -> str:
        """Build prompt for emotion recognition."""
        emotions_str = ", ".join(emotions_list)
        
        if self.prompt_name == "user_labels":
            prompt = f"Classify the emotion in this speech as one of: {emotions_str}. Respond with only the emotion label. Answer:"
        elif self.prompt_name == "detailed":
            prompt = f"Analyze the emotion in this speech and classify it as one of: {emotions_str}. Provide a brief explanation."
        else:
            prompt = f"What emotion is expressed in this speech? Options: {emotions_str}"
        
        return prompt

    def collate_fn(self, inputs):
        input_audios = [_['audio'] for _ in inputs]
        labels = [_['label'] for _ in inputs]
        keys = [_.get('key', '') for _ in inputs]

        return {
            'audio': input_audios,
            'labels': labels,
            'keys': keys
        }

    def predict(self, inputs: dict) -> List[Union[str, int]]:
        """Generate predictions for a batch of inputs."""
        input_audios = inputs['audio']

        prompt = self._build_prompt(self.class_labels)
        
        outputs = []
        with torch.no_grad():
            for audio in input_audios:
                try:
                    wav_path = audio

                    output = self.model.generate(
                        wav_path=wav_path,
                        prompt=prompt,
                        device=self.device,
                        max_length=self.max_new_tokens,
                        num_beams=self.num_beams,
                        do_sample=self.do_sample,
                        top_p=self.top_p,
                        temperature=self.temperature,
                    )
                    
                    outputs.append(output[0] if isinstance(output, list) else output)
                except Exception as e:
                    logger.error(f"Error processing audio: {e}")
                    outputs.append("Unknown")
        
        predictions = postprocess_ser_response(
            class_labels=self.class_labels,
            model_responses=outputs,
        )
        return predictions

    def forward(self, inputs: dict):
        pass