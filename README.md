# Pyton-Generator-PHPAiModel-RNN

## Overview
This project is a **Python port** of [PHPAiModel-RNN](https://github.com/iStark/PHPAiModel-RNN).  
It implements a **simple Recurrent Neural Network (RNN)** with tanh activations and softmax output, trained via **Backpropagation Through Time (BPTT)** and **Adagrad optimizer**.  

- Written fully in **Python 3** (Flask + NumPy).  
- Provides a **web interface** (like the PHP version) to train RNN models on custom datasets.  
- Trains models on **RU/EN datasets** with `<Q> question`, `<A> answer`, and `<NL>` markers.  
- Saves trained models into `/Models` as JSON.  

---

## Features
- Tokenization with `<BOS>`, `<EOS>`, `<NL>` markers  
- Vocabulary building (token ↔ id)  
- Tanh-based hidden state RNN  
- Cross-entropy loss  
- Training with **Adagrad** optimizer  
- Real-time training logs (progress %, ETA, average loss)  
- Web UI with:
  - Dataset selection  
  - Hyperparameters: hidden size, seq length, epochs, learning rate  
  - Progress logs and model listing  

---

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourname/Pyton-Generator-PHPAiModel-RNN
   cd Pyton-Generator-PHPAiModel-RNN
   pip install flask numpy
   ```
   
   
---

## Project Structure
Python-Generator-PHPAiModel-RNN/
│
├── app.py              # Main Flask app with RNN trainer + web UI
├── Datasets/           # Training datasets (*.txt with Q/A/ NL markup)
├── Models/             # Trained RNN models (JSON format)
└── README.md           # Project documentation

---
## Example Dataset Format

Each dataset should follow Q/A with markers :

<Q> What's your name? <A> I'm Bot. Nice to meet you! <NL>
<Q> Hi <A> Hello <NL>

---
## Model Output

Trained models are stored as `.json` in `/Models`:

- `Wxh`, `Whh`, `Why`, `bh`, `by` matrices  
- `vocab` / `ivocab` mappings  
- `metadata`: dataset, epochs, seq length, learning rate, training time, token count  

---

## License

MIT License © 2025  
Developed by **Artur Strazewicz**

---

## Links
- Original PHP Project: [PHPAiModel-RNN](https://github.com/iStark/PHPAiModel-RNN)  
- Twitter/X: [@strazewicz](https://x.com/strazewicz)  
- TruthSocial: [@strazewicz](https://truthsocial.com/@strazewicz)
