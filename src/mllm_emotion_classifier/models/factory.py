from .qwen2_audio import Qwen2AudioEmotionWrapper
from .qwen2_audio_instruct import Qwen2AudioInstructEmotionWrapper
try:
    from .audioflamingo3_instruct import AudioFlamingo3EmotionWrapper
except:
    # AudioFlamingo3EmotionWrapper = None
    pass
from .voxtral import VoxtralEmotionWrapper
from .salmonn import SALMONNEmotionWrapper
from .base import BaseEmotionModel

class ModelFactory:
    def create(name: str, **kwargs) -> BaseEmotionModel:
        if name == "qwen2-audio":
            model = Qwen2AudioEmotionWrapper(checkpoint="Qwen/Qwen2-Audio-7B", **kwargs)
        elif name == "qwen2-audio-instruct":
            model = Qwen2AudioInstructEmotionWrapper(checkpoint="Qwen/Qwen2-Audio-7B-Instruct", **kwargs)
        elif name == "audio-flamingo-3":
            model = AudioFlamingo3EmotionWrapper(**kwargs)
        elif name == "voxtral-mini":
            model = VoxtralEmotionWrapper(**kwargs)
        elif name == "salmonn-7b":
            model = SALMONNEmotionWrapper(
                ckpt='SALMONN_7B/salmonn_7b_v0.pth',
                whisper_path='openai/whisper-large-v2',
                beats_path='SALMONN_7B/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt',
                vicuna_path='lmsys/vicuna-7b-v1.5',
                **kwargs
            )
        else:
            raise ValueError(f"Model '{name}' not found.")
        return model