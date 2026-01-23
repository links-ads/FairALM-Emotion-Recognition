"""Chat prompts for AudioFlamingo3 Emotion Recognition."""

import os

CHAT_PROMPTS = {
    "system_labels": {
        "system": "You are a highly capable assistant specialized in Speech Emotion Recognition. You receive audio input and identify the emotion expressed in the speech. The emotion must be one of the following categories: {labels}.",
        "user": "Classify the tone of the speaker in the preceding audio. Answer:"
    },
    "user_labels": {
        "system": "You are a highly capable assistant specialized in Speech Emotion Recognition",
        "user": "Classify the speaker's tone in the preceding audio. Options: {labels}. Answer:"
    },
    "direct": {
        "system": None,
        "user": "Listen to the audio and identify the speaker's emotion. Choose from: {labels}. Answer:"
    }
}


def get_chat_prompt(prompt_name: str) -> dict:
    """Get the chat prompt template by name."""
    return CHAT_PROMPTS.get(prompt_name, CHAT_PROMPTS["user_labels"])


def build_conversation(prompt_name: str, labels_str: str, audio_path: str = None, audio_format: str = "path") -> list:
    prompt = get_chat_prompt(prompt_name)
    
    user_text = prompt["user"].format(labels=labels_str) if "{labels}" in prompt["user"] else prompt["user"]
    
    if audio_format == "path":
        file_url = "file://" + os.path.abspath(audio_path)
    elif audio_format == "array":
        file_url = 'placeholder'

    content = [
        {'type': 'text', 'text': user_text},
        {'type': 'audio', 'path': file_url}
    ]
    
    if prompt["system"] is not None:
        system_text = prompt["system"].format(labels=labels_str) if "{labels}" in prompt["system"] else prompt["system"]
        content.insert(0, {'type': 'text', 'text': system_text})
    
    conversation = [
        {
            'role': 'user',
            'content': content
        }
    ]
    
    return conversation