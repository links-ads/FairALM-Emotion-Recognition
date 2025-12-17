import pandas as pd
from base import FairnessAnnotator
from pathlib import Path


LANGUAGE = "English"
SPEAKER_INFO = {
    "bea": {
        "gender": "Female"
    },
    "jenie": {
        "gender": "Female"
    },
    "josh": {
        "gender": "Male"
    },
    "sam": {
        "gender": "Male"
    }
}

class EMOVDBFairnessAnnotator(FairnessAnnotator):
    """Fairness preprocessor for EMOVDB dataset."""
    
    def __init__(self):
        super().__init__(dataset_name="emovdb", num_folds=1)
        self.LANGUAGE = LANGUAGE
    
    def _extract_gender_from_key(self, key: str) -> str:
        """
        Extract gender from EMOVDB key (e.g., 'emovdb-sam-Amused-0384').

        Parameters:
            key (str): The key string from which to extract gender.

        Returns:
            str: The gender of the speaker ('Male' or 'Female').
        """
        speaker = key.split('-')[1]
        return SPEAKER_INFO.get(speaker, {}).get('gender', 'Unknown')
    
    def extract_sensitive_attributes(self, entry: dict) -> dict:
        """Extract sensitive attributes for EMOVDB."""
        key = entry.get('key', '')
        return {
            'gender': self._extract_gender_from_key(key),
            'language': self.LANGUAGE,
        }

if __name__ == "__main__":
    annotator = EMOVDBFairnessAnnotator()
    annotator.add_sensitive_attributes_to_all_folds()