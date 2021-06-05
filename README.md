# Neural_Machine_Translation

## Best model metrics
| Metric          | Score                   |
| ----------------|:-----------------------:|
|  Corpus BLEU    | 37.0347                 |
|  Dev ppl        | 61.4084                 |
You can find the model weights [here](https://www.dropbox.com/s/etkwvmnqfo26hrc/model.bin?dl=0)
--------------------------------------------------------------------------------------------
## Try Translating yourself!
- The translation demo is available [here](https://share.streamlit.io/sahilkhose/neural_machine_translation/main/stream_translate.py) on Streamlit Sharing.
- Even is you don't know Spanish you can use the demo as there is Google Translate which will help you to convert your English sentences to Spanish.
--------------------------------------------------------------------------------------------

## About NMT model
## Hybrid Word-Character Seq2Seq Machine Translation
- It is a Seq2Seq Model that translates Spanish sentences into English based on [Luong et al. 2015](https://arxiv.org/pdf/1508.04025.pdf). 
- It consists of a bidirectional LSTM encoder and unidirectional LSTM decoder.
- It also uses attention mechanism to boost its performance on the translation task.
- The pipeline and the implementations is inspired by the [Open-NMT](https://github.com/OpenNMT/OpenNMT-py) package. 

<p align="center">
<img src="https://github.com/sahilkhose/Neural_Machine_Translation/blob/master/figures/nmt.png" alt="drawing" width="350"/>
</p>

- The model becomes more powerful as we combine character-level with word-level language modelling. 
- The idea is that whenever the NMT model generates a \<unk> token we run a character-level language model and generate a word in the output character by character. 
- This hybrid word-character approach was proposed by [Luong and Manning 2016](https://arxiv.org/pdf/1604.00788.pdf) and turned out to be effective in increasing the performance of the NMT model (+1.2 BLEU).
<p align="center">
<img src="https://github.com/sahilkhose/Neural_Machine_Translation/blob/master/figures/nmt-hybrid.png" alt="drawing" width="350"/>
</p>

--------------------------------------------------------------------------------------------


## Installation

Install from source:
```bash
git clone https://github.com/sahilkhose/Neural_Machine_Translation
cd Neural_Machine_Translation
pip3 install -r requirements.txt
```
To run the translation demo:
```bash
streamlit run stream_translate.py
```
Or just go [here](https://share.streamlit.io/sahilkhose/neural_machine_translation/main/stream_translate.py) on Streamlit Sharing.

--------------------------------------------------------------------------------------------

## Contributing

If you find a bug, create a GitHub issue, or even better, submit a pull request. Similarly, if you have questions, simply post them as GitHub issues.
