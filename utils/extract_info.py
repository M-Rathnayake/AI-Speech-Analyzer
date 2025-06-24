#info extraction
import re
import spacy
from typing import Dict, List
from dataclasses import dataclass

@dataclass
class ExtractedInfo:
    phones: List[str]
    emails: List[str]
    names: List[str]

class InfoExtractor:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
        self.phone_regex = re.compile(r'[\+\(]?[0-9][0-9\-\(\)\s]{8,}[0-9]')
        self.email_regex = re.compile(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}')

    def extract(self, text: str) -> ExtractedInfo:
        """Extract structured info from text.
        
        Args:
            text: Input text to analyze
            
        Returns:
            ExtractedInfo dataclass with results
        """
        phones = self.phone_regex.findall(text)
        emails = self.email_regex.findall(text)
        doc = self.nlp(text)
        names = [ent.text for ent in doc.ents if ent.label_ == "PERSON"]
        
        return ExtractedInfo(
            phones=phones,
            emails=emails,
            names=names
        )

# Example usage
if __name__ == "__main__":
    extractor = InfoExtractor()
    info = extractor.extract("Call (123) 456-7890 or email john@example.com")
    print(info)