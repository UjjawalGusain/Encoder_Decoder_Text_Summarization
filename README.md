
# LSTM-Based Encoder-Decoder Text Summarizer Built from Scratch

This project is a text summarization tool built entirely from scratch using an LSTM-based encoder-decoder architecture. While it has been trained on a toy dataset and is limited in performance, it is correctly implemented and demonstrates the complete process of building a summarization model. The results are constrained by available resources rather than the implementation itself.


## Contents

- Project Overview
- Features
- Usage
    - Training the Model
    - Generating Summaries
- Model Details
- Limitations
- License
- Acknowledgements



## Features

- End-to-End Text Summarization: Implements an encoder-decoder architecture using LSTM for generating summaries from input text.

- Custom Model Design: Built from scratch with no reliance on pre-trained models, showcasing a fundamental approach to sequence-to-sequence tasks.

- Training on Toy Dataset: Trained on a small dataset to demonstrate functionality, with potential for scaling to larger datasets.

- Hands-on Understanding: Provides an educational resource for understanding encoder-decoder models, LSTM, and the summarization process.
## Usage

### Training the Model
To train the model, simply run the provided code in the Jupyter notebook. The model is trained using an encoder-decoder architecture with LSTM on a toy dataset. Follow the steps below to begin training:

- Load and preprocess your dataset.
- Define the encoder and decoder models.
- Train the model by running the training loop with the specified number of epochs.
- Monitor the training process using the loss metrics and adjust hyperparameters as needed.

### Generating Summaries
Once the model is trained, use the provided generate_summary() function to generate summaries for input text. The function requires the trained model, tokenizers (word2idx, idx2word), and a maximum length for the generated summary. Pass the input text to get a concise summary as output.

Example:
```
generated_summary = generate_summary(model, input_text, word2idx, idx2word, max_length=50)
print("Generated Summary: ", " ".join(generated_summary))
```
## Architecture

The model follows an Encoder-Decoder architecture using LSTM (Long Short-Term Memory) networks to perform text summarization. The architecture consists of the following components:

### Encoder
- The encoder processes the input text, tokenized into sequences, and encodes it into a fixed-size context vector.
- It uses a stack of LSTM layers to capture the dependencies between words in the input sequence.

### Decoder
- The decoder generates the summary from the context vector produced by the encoder.
- Similar to the encoder, the decoder also uses LSTM layers, taking the context vector and a previous word (during training) to predict the next word in the summary.
- The decoder ends with a softmax layer to output probabilities for each word in the vocabulary.

Architecture Diagram:
![encoder-decoder_2 drawio](https://github.com/user-attachments/assets/65fb2000-cea9-49fa-824a-140619f24866)



## Acknowledgement
`Programming isn't about what you know; it's about what you can figure out.` ~Chris Pine

A huge thank you to the video tutorials and research papers that helped me along the way. While the architecture I implemented is not original, the code is entirely self-written.

This project is an educational tool, and I recognize that many industry-level solutions now exist with state-of-the-art results and higher abstraction.

The goal of this project was to learn and appreciate the beauty of neural networks and mathematics in general.
