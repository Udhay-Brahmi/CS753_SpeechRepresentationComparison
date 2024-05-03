# Speech Representation Comparison

This project compares various speech representation models on the LibriSpeech dataset's test split audio files. We evaluate the representations from several pre-trained models including Wav2Vec 2.0 (base and large), HuBERT (base and large), and XLS-R (large). Our baseline for comparison is the HuBERT large model, and we utilize Centered Kernel Alignment (CKA) as the metric for comparison. Additionally, results are visualized through heat maps.

## Models Compared
- **Wav2Vec 2.0 Base**: Pre-trained model for speech recognition.
- **Wav2Vec 2.0 Large**: A larger variant of Wav2Vec 2.0, expected to capture more detailed features.
- **HuBERT Base**: A base model using hidden-unit BERT for speech representation.
- **XLSR Large**: Cross-lingual speech representation model, trained on multiple languages.
- **HuBERT Large**: Serves as the baseline for comparison; known for its robust performance in various speech tasks.

## Metric
- **Centered Kernel Alignment (CKA)**: Used to measure the similarity between the representations from different models. CKA code derived from [this repository.](https://github.com/yuanli2333/CKA-Centered-Kernel-Alignment/tree/master)

## Prerequisites
- Python 3.8 or newer
- PyTorch
- Transformers library
- Numpy

## Installation
```bash
git clone https://github.com/Udhay-Brahmi/CS753_SpeechRepresentationComparison.git
cd CS753_SpeechRepresentationComparison
pip install -r requirements.txt
```

![download (2)](https://github.com/Udhay-Brahmi/CS753_SpeechRepresentationComparison/assets/72250606/fd25644a-852f-4b1f-bb88-86a107697dca)
![download (3)](https://github.com/Udhay-Brahmi/CS753_SpeechRepresentationComparison/assets/72250606/e7f9e7d3-81be-4c11-8126-423840f2973c)
![download (5)](https://github.com/Udhay-Brahmi/CS753_SpeechRepresentationComparison/assets/72250606/ea6cd677-a8a6-48b4-8ac1-30fd0858ac4c)
![download (4)](https://github.com/Udhay-Brahmi/CS753_SpeechRepresentationComparison/assets/72250606/db69986a-675b-44f4-8ca8-ff1966d12b26)
