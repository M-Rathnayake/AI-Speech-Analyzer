# 🎤 AI Speech Analyzer
This project was built as a learning exercise - while functional, it has clear areas for growth. I've documented these limitations transparently to guide future iterations and help others learn from my experience

An end-to-end system that transcribes audio, extracts key information (phone numbers, emails, names), and answers natural language questions about the content.

![image](https://github.com/user-attachments/assets/d8796faa-4a8b-47c8-8012-a615fc30e604)

## Features

- **Accurate Speech-to-Text**  
  Powered by OpenAI's Whisper model with 80%+ accuracy on clear audio
- **Smart Information Extraction**  
  Identifies and categorizes:
  -  Phone numbers (including international formats)
  -  Email addresses
  -  Person names
- **Question Answering**  
  Answers natural language questions about the transcript
- **User-Friendly Interface**  
  Simple web app built with Streamlit

## Technical Stack

| Component          | Technology Used |
|--------------------|-----------------|
| Speech Recognition | OpenAI Whisper  |
| NLP Processing     | SpaCy           |
| Question Answering | Haystack        |
| Web Interface      | Streamlit       |
| Deployment Ready   | Docker          |

## Limitations
1. Audio Quality Sensitivity
   - Struggles with heavy accents, background noise, or overlapping speech
2. Language Support
   - Primarily optimized for English (though Whisper supports 50+ languages)
   - named entity recognition(SpaCy) works best for western names
3. Model Knowledge Cutoff
   - QA system answers only from transcribed content (no external knowledge)
4. Hardware Dependencies
   - No GPU acceleration implemented (CPU-only processing)
5. performance constraints
   - Large audio files (>10 mins) may cause memory issues
   - does not support live microphone input
     
## Imrpovements
1. Accuracy Enhancements
   - Add audio pre-processing (noise reduction, volume normalization)
2.Performance Upgrades
   - Add chunked processing for long recordings
3. User Experience 
   - Real-time transcription feedback
4. technical Debt
   - Write unit tests (currently 0% test coverage)
5. Accessibility 
   -Support for live microphone input
   - Export results as CSV/PDF
     
### Prerequisites
- Python 3.10+  (for this project I used python 3.11.1)
- FFmpeg (for audio processing)
