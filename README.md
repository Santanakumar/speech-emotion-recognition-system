# speech-emotion-recognition-system
Speech Emotion Recognition system with Speaker Diarization and Emotion Analysis for Enhanced multi-speaker conversations.

## EXECUTIVE SUMMARY
The project "Speech Emotion Recognition with Speaker Diarization and Emotion Analysis for Enhanced Conversation" presents a real-time AI framework designed to detect and interpret emotions in spoken conversations. Traditional sentiment analysis approaches rely mainly on text input and often fail to capture important emotional cues embedded in speech, such as tone, pitch, and prosody.

To address these limitations, the proposed system integrates "Speech Emotion Recognition (SER)" with "Speaker Diarization" and "Explainable AI (XAI)" to deliver an emotionally aware and context-sensitive solution. The framework leverages advanced machine learning models, including "OpenAI Whisper" for robust multilingual speech-to-text transcription, "PyAnnote" for accurate speaker segmentation, and transformer-based NLP models such as "RoBERTa" and "DistilBERT" for sentiment-aware emotion classification.

A key strength of the system is its ability to process "multi-speaker audio", associate detected emotions with specific speakers, and provide real-time visual feedback through a "Flask-based web interface". Emotions such as happiness, sadness, anger, and neutrality are identified along with confidence scores, while XAI components enhance transparency by presenting emotional trends and speaker timelines.

Through optimized preprocessing, parallelized execution, and GPU acceleration, the system demonstrates strong real-time performance, scalability, and adaptability across languages and domains. The project serves as both a functional prototype and a research contribution, offering a comprehensive approach toward building emotionally intelligent AI systems for applications such as mental health monitoring, virtual education, and AI- driven customer support.

## INTRODUCTION
### Background

Speech Emotion Recognition (SER), combined with real-time sentiment analysis, plays a key role in building emotionally intelligent AI systems for domains such as healthcare, customer support, education, and human-computer interaction. Traditional sentiment analysis methods are mostly text-based and often fail to capture critical cues present in speech, such as tone, pitch, and prosody.

To overcome this limitation, the project introduces a real-time framework that integrates both speech and text-based emotion analysis. The system leverages "OpenAI Whisper" for accurate multilingual speech transcription, "PyAnnote" for speaker diarization in multi-speaker conversations, and transformer-based NLP models such as "RoBERTa" and "DistilBERT" for sentiment-aware emotion classification. Unlike conventional SER approaches that depend on handcrafted audio features (MFCCs, spectrograms), this solution adopts an end-to-end deep learning pipeline for improved adaptability and contextual understanding.

### Motivation

The motivation behind this project stems from the growing need for AI systems that can understand and respond to human in real time. Text-only analysis often leads to incomplet emotional interpretation, reducing the effectiveness of conversational agents, mental health tools, and virtual assistants.

By enabling emotion-aware speech processing, the system can support sensitive applications such as early distress detection in mental health monitoring, empathetic customer service automation, and more natural human-like interactions in AI assistants. Additionally, Explainable AI (XAI) components provide confidence scores and emotion visualizations, improving transparency and user trust.

### Scope of the Project
The project focuses on developing a scalable and real-time SER framework capable of:

- Detecting emotions from spoken conversations.
- Handling multi-speaker audio through speaker diarization.
- Performing multilingual transcription and sentiment-aware emotion classification.
- Providing speaker-specific emotion timelines and visual insights via a Flask-based interface.

The system is designed for real-world deployment in areas such as healthcare support, virtual education, intelligent tutoring systems, and AI-driven customer engagement platforms. By combining speech processing, transformer-based emotion inference, and explainable feedback, the project aims to advance emotionally intelligent and human-centric AI communication.
