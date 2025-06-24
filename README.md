# 🎤 AI Speech Analyzer

An end-to-end system that transcribes audio, extracts key information (phone numbers, emails, names), and answers natural language questions about the content.

![Demo Screenshot](demo.gif) *(Add a screenshot or screen recording later)*

## Features

- **Accurate Speech-to-Text**  
  Powered by OpenAI's Whisper model with 90%+ accuracy on clear audio
- **Smart Information Extraction**  
  Identifies and categorizes:
  - ☎️ Phone numbers (including international formats)
  - ✉️ Email addresses
  - 👤 Person names
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

## Quick Start

### Prerequisites
- Python 3.10+  (for this project I used python 3.11.1)
- FFmpeg (for audio processing)
