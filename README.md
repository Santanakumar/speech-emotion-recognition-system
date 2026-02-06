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

## RESEARCH GAPS ADDRESSED
Despite significant progress in Speech Emotion Recognition (SER) and sentiment analysis, several limitations still reduce the real-world effectiveness of current systems. The project aims to bridge these gaps through an advanced, real-time, and explainable framework.

### Key Gaps: Identified & Proposed Solutions

- **Limited Context Awareness:** Most SER models analyze utterances in isolation, failing to capture emotional flow across conversational turns.
  *Solution:* Transformer-based models (RoBERTa, DistilBERT) are integrated with diarized speech to preserve dialogue continuity.

- **Dependence on Handcrafted Features:** Traditional approaches rely heavily on MFCCs and spectrograms, which are sensitive to noise, accents, and tuning bias.
  *Solution:* End-to-end deep learning representations (e.g., HuBERT) reduce feature engineering dependency and improve generalization.

- **Real-time Processing Latency:** Deep learning SER pipelines often suffer from high computational cost, limiting live deployment.
  *Solution:* Optimized parallel execution enables simultaneous transcription, diarization, and emotion interference for low-latency performance.

- **Lack of Multimodal Emotion Fusion:** Many systems process only speech or text independently, missing complementary emotional cues.
  *Solution:* This framework combines prosodic speech signals with semantic text understanding for richer emotion classification.

- **Language and Accent Variability:** Existing SER models underperform across multilingual and diverse accent scenarios due to dataset bias.
  *Solution:* Whisper's multilingual transcription and HuBERT's robust speech embeddings enhance cross-lingual adaptability.

- **Emotion Ambiguity and Overlap:** Emotions such as anger, frustration, or excitement often overlap, making classification difficult.
  *Solution:* Joint speech-text modeling improves disambiguation of subtle emotional expressions.

- **Lack of Explainability:** Most SER systems provide predictions without transparency, reducing trust in sensitive applications.
  *Solution:* Explainable AI (XAI) modules visualize confidence scores, emotion trends, and speaker timelines for interpretability.

By addressing these gaps, the project delivers a scalable, context-aware, multilingual, and explainable SER framework suitable for real-world human-centric AI applications.

## PROBLEM STATEMENT
Speech is one of the richest forms of human communication, conveying not only words but also emotional cues through tone, pitch, and prosody. However, most traditional sentiment analysis systems rely maninly on text input and fail to capture these acoustic emotional dimensions, leading to incomplete or inaccurate emotion interpretation.

Existing Speech Emotion Recognition (SER) models face several challenges, including:

- Limited contextual understanding across conversational turns
- Difficulty handling multi-speaker conversations
- High latency in real-time applications
- Poor adaptability to multilingual speech and accent variations
- Lack of transparency due to black-box deep learning predictions

To address these limitations, this project proposes a **real-time, explainable Speech Emotion Recognition framework** that combines speech transcription, speaker diarization, transformer-based emotion classification, and emotion visualization to enhance emotionally intelligent AI interactions.

## OBJECTIVES
The primary objective of this project is to design and implement a **real-time Speech Emotion Recognition (SER) system** integrated with sentiment-aware analysis to improve the emotional intelligence of AI-driven conversational applications.

### Key Objective Include:
- **Real-Time Emotion Detection:** Develop an end-to-end pipeline capable of detecting emotions from spoken conversations with low latency.
- **Multilingual Speech Transcription:** Use **OpenAI Whisper** to generate accurate speech-to-text transcriptions even undder noisy conditions and diverse accents.
- **Speaker-Level Emotion Association:** Apply **PyAnnote Speaker Diarization** to separate multiple spekers and map emotions to individual voices.
- **Context-Aware Emotion Classification:** Leverage transformer-based NLP models (**RoBERTa, DistilBERT**) to classify emotions such as happiness, sadness, anger, fear and neutrality.
- **Explainable AI Integration (XAI):** Provide interpretable outputs through confidence scores, emotion timelines, and real-time visual feedback for improved trust.

By achieving these objectives, the project delivers a scalable, context-aware, and emotionally intelligent AI framework that enhances human-computer interaction.

## Project Plan & Development Phases
The project was executed in a structured multi-phase workflow to ensure scalability, accuracy, and real-time performance.

### Phase 1: Research & Requirement Analysis
- Reviewed existing work in Speech Emotion Recognition (SER), speaker diarization, and sentiment analysis.
- Identified key challenges such as real-time latency, lack of interpretability, and speaker-level emotion attribution.
- Defined functional and non-functional requirements.
- Finalized tools: Python, Whisper, PyAnnote, Transformers and Flask.

### Phase 2: Data Collection & Preprocessing
