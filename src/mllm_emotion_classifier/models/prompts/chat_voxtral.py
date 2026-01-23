"""Chat prompts for Voxtral Emotion Recognition."""

import os

CHAT_PROMPTS = {
    "system_labels": {
        "system": None,  # System messages not allowed with audio
        "user": "You are a highly capable assistant specialized in Speech Emotion Recognition. Classify the tone of the speaker in the preceding audio. The emotion must be one of the following categories: {labels}. Answer:"
    },
    "user_labels": {
        "system": None,  # System messages not allowed with audio
        "user": "You are an emotion classifier.\nPossible emotions: {labels}.\nFrom the given audio, classify the emotion. Answer:"
    },
    "direct": {
        "system": None,
        "user": "Listen to the audio and identify the speaker's emotion. Choose from: {labels}. Answer:"
    }
}


def get_chat_prompt(prompt_name: str) -> dict:
    """Get the chat prompt template by name."""
    return CHAT_PROMPTS.get(prompt_name, CHAT_PROMPTS["user_labels"])


def build_conversation(prompt_name: str, labels_str: str, audio_input: str, audio_format: str = "base64") -> list:
    """Build conversation for Voxtral format.
    
    Voxtral expects format (NO system message when audio is present):
    [
        {
            "role": "user",
            "content": [
                {"type": "audio", "url": "..."} or
                {"type": "audio", "path": "..."} or
                {"type": "audio", "base64": "..."},
                {"type": "text", "text": "question"}
            ]
        }
    ]
    
    Args:
        prompt_name: Name of the prompt template
        labels_str: Comma-separated string of emotion labels
        audio_input: Audio data (path, URL, or base64 string)
        audio_format: One of "path", "url", or "base64"
    """
    prompt = get_chat_prompt(prompt_name)
    
    user_text = prompt["user"].format(labels=labels_str) if "{labels}" in prompt["user"] else prompt["user"]
    
    # Build audio content based on format
    if audio_format == "path":
        audio_content = {"type": "audio", "path": audio_input}
    elif audio_format == "url":
        file_url = "file://" + os.path.abspath(audio_input)
        audio_content = {"type": "audio", "url": file_url}
    elif audio_format == "base64":
        audio_content = {"type": "audio", "base64": audio_input}
    else:
        raise ValueError(f"Unsupported audio_format: {audio_format}. Must be 'path', 'url', or 'base64'.")
    
    # Build content list with audio first, then text
    content = [
        audio_content,
        {
            "type": "text",
            "text": user_text,
        },
    ]
    
    # Build conversation with only user message (no system message allowed with audio)
    conversation = [
        {
            "role": "user",
            "content": content
        }
    ]
    
    return conversation