import pandas as pd
from base import FairnessAnnotator
from pathlib import Path


LANGUAGE = "English"
SPEAKER_DEMOGRAPHICS_FILE_PATH = "downloads/emns/cleaned_webm/metadata.csv"

class EMNSFairnessAnnotator(FairnessAnnotator):
    """Fairness preprocessor for EMNS dataset."""
    
    def __init__(self):
        super().__init__(dataset_name="emns", num_folds=1)
        self.demographics = self._read_demographics_file()
        self.LANGUAGE = LANGUAGE

    def _read_demographics_file(self):
        """
        Reads the demographics CSV file and returns a dictionary mapping audio file name to their demographics. The file is composed of the following columns: id|utterance|description|emotion|date_created|status|gender|age|level|audio_recording|user_id. It is separated by '|'.

        Returns:
            dict: A dictionary where keys are audio file names and values are dictionaries of demographics (i.e. gender, age).
        """
        df = pd.read_csv(SPEAKER_DEMOGRAPHICS_FILE_PATH, sep='|')
        demographics_dict = {}
        for _, row in df.iterrows():
            audio_recording = row['audio_recording']
            audio_recording = Path(audio_recording).stem
            audio_recording = f"emns-{audio_recording}"
            demographics_dict[audio_recording] = {
                'Gender': row['gender'],
                'Age': row['age'],
            }
        return demographics_dict
    
    def _extract_gender_from_key(self, key: str) -> str:
        """
        Extract gender from EMNS key (e.g., 'recorded_audio_0bJpmSM').
        
        The first four digits represent the audio file name. The information about demographics and sensityve attributes is stored in a separate CSV file.

        Parameters:
            key (str): The key string from which to extract gender.

        Returns:
            str: The gender of the speaker ('Male' or 'Female').
        """
        demographics = self.demographics.get(key, {})
        return demographics.get('Gender', 'Unknown')
    
    def _extract_age_from_key(self, key: str) -> str:
        """
        Extract age from EMNS key (e.g., 'recorded_audio_0bJpmSM').
        
        The first four digits represent the audio file name. The information about demographics and sensityve attributes is stored in a separate CSV file.

        Parameters:
            key (str): The key string from which to extract age.

        Returns:
            str: The age of the speaker.
        """
        demographics = self.demographics.get(key, {})
        return demographics.get('Age', 'Unknown')
    
    def extract_sensitive_attributes(self, entry: dict) -> dict:
        """Extract sensitive attributes for EMNS."""
        key = entry.get('key', '')
        return {
            'gender': self._extract_gender_from_key(key),
            'age': self._extract_age_from_key(key),
            'language': self.LANGUAGE,
        }

if __name__ == "__main__":
    annotator = EMNSFairnessAnnotator()
    annotator.add_sensitive_attributes_to_all_folds()