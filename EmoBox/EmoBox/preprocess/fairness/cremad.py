import pandas as pd
from base import FairnessAnnotator


LANGUAGE = "English"
SPEAKER_DEMOGRAPHICS_FILE_PATH = "downloads/cremad/VideoDemographics.csv"
SPEAKER_ID_START_INDEX = 7
SPEAKER_ID_END_INDEX = 11

class CREMADFairnessAnnotator(FairnessAnnotator):
    """Fairness preprocessor for CREMA-D dataset."""
    
    def __init__(self):
        super().__init__(dataset_name="cremad", num_folds=1)
        self.demographics = self._read_demographics_file()
        self.LANGUAGE = LANGUAGE

    def _read_demographics_file(self):
        """
        Reads the demographics CSV file and returns a dictionary mapping speaker IDs to their demographics.

        Returns:
            dict: A dictionary where keys are speaker IDs and values are dictionaries of demographics.
        """
        df = pd.read_csv(SPEAKER_DEMOGRAPHICS_FILE_PATH)
        demographics_dict = {}
        for _, row in df.iterrows():
            speaker_id = str(row['ActorID']).zfill(4)  # Ensure speaker ID is zero-padded to 4 digits
            demographics_dict[speaker_id] = {
                'Gender': row['Sex'],
                'Age': row['Age'],
                'Race': row['Race'],
                'Ethnicity': row['Ethnicity']
            }
        return demographics_dict

    
    def _extract_gender_from_key(self, key: str) -> str:
        """
        Extract gender from CREMA-D key (e.g., 'cremad-1001_DFA_ANG_XX').
        
        The first four digits represent the speaker ID. The information about demographics and sensityve attributes is stored in a separate CSV file.

        Parameters:
            key (str): The key string from which to extract gender.

        Returns:
            str: The gender of the speaker ('Male' or 'Female').
        """
        speaker_id = key[SPEAKER_ID_START_INDEX:SPEAKER_ID_END_INDEX]
        demographics = self.demographics.get(speaker_id, {})
        return demographics.get('Gender', 'Unknown')
    
    def _extract_age_from_key(self, key: str) -> str:
        """
        Extract age from CREMA-D key (e.g., 'cremad-1001_DFA_ANG_XX').
        
        The first four digits represent the speaker ID. The information about demographics and sensityve attributes is stored in a separate CSV file.

        Parameters:
            key (str): The key string from which to extract age.

        Returns:
            str: The age of the speaker.
        """
        speaker_id = key[SPEAKER_ID_START_INDEX:SPEAKER_ID_END_INDEX]
        demographics = self.demographics.get(speaker_id, {})
        return demographics.get('Age', 'Unknown')
    
    def _extract_race_from_key(self, key: str) -> str:
        """
        Extract race from CREMA-D key (e.g., 'cremad-1001_DFA_ANG_XX').
        
        The first four digits represent the speaker ID. The information about demographics and sensityve attributes is stored in a separate CSV file.

        Parameters:
            key (str): The key string from which to extract race.

        Returns:
            str: The race of the speaker.
        """
        speaker_id = key[SPEAKER_ID_START_INDEX:SPEAKER_ID_END_INDEX]
        demographics = self.demographics.get(speaker_id, {})
        return demographics.get('Race', 'Unknown')
    
    def _extract_ethnicity_from_key(self, key: str) -> str:
        """
        Extract ethnicity from CREMA-D key (e.g., 'cremad-1001_DFA_ANG_XX').
        
        The first four digits represent the speaker ID. The information about demographics and sensityve attributes is stored in a separate CSV file.

        Parameters:
            key (str): The key string from which to extract ethnicity.

        Returns:
            str: The ethnicity of the speaker.
        """
        speaker_id = key[SPEAKER_ID_START_INDEX:SPEAKER_ID_END_INDEX]
        demographics = self.demographics.get(speaker_id, {})
        return demographics.get('Ethnicity', 'Unknown')
    
    def extract_sensitive_attributes(self, entry: dict) -> dict:
        """Extract sensitive attributes for CREMA-D."""
        key = entry.get('key', '')
        return {
            'gender': self._extract_gender_from_key(key),
            'age': self._extract_age_from_key(key),
            'race': self._extract_race_from_key(key),
            'ethnicity': self._extract_ethnicity_from_key(key),
            'language': self.LANGUAGE,
        }

if __name__ == "__main__":
    annotator = CREMADFairnessAnnotator()
    annotator.add_sensitive_attributes_to_all_folds()