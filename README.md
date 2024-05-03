# CS753_SpeechRepresentationComparison

# Speech Representation Comparison

This project compares various speech representation models on the LibriSpeech dataset's test split audio files. We evaluate the representations from several pre-trained models including Wav2Vec 2.0 (base and large), HuBERT (base and large), and XLS-R (large). Our baseline for comparison is the HuBERT large model, and we utilize Centered Kernel Alignment (CKA) as the metric for comparison. Additionally, results are visualized through heat maps.

## Models Compared
- **Wav2Vec 2.0 Base**: Pre-trained model for speech recognition.
- **Wav2Vec 2.0 Large**: A larger variant of Wav2Vec 2.0, expected to capture more detailed features.
- **HuBERT Base**: A base model using hidden-unit BERT for speech representation.
- **XLSR Large**: Cross-lingual speech representation model, trained on multiple languages.
- **HuBERT Large**: Serves as the baseline for comparison; known for its robust performance in various speech tasks.

## Metric
- **Centered Kernel Alignment (CKA)**: Used to measure the similarity between the representations from different models.

## Prerequisites
- Python 3.8 or newer
- PyTorch
- Transformers library
- Numpy
- Matplotlib
