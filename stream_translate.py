import streamlit as st
from google_trans_new.google_trans_new import google_translator
from docopt import docopt
import run
import numpy as np 
import sys
import torch
import re 
import nltk
import os

@st.cache
def download_data():
    
    path1 = 'model.bin'
    if not os.path.exists(path1):
        decoder_url = 'wget -O model.bin https://www.dropbox.com/s/etkwvmnqfo26hrc/model.bin?dl=0'
        
        with st.spinner('Done!\nModel weights were not found, Downloading them...'):
            os.system(decoder_url)
    else:
        print("Model is here.")

@st.cache
def load_model(args):
    nltk.download('punkt')
    model = run.NMT.load(args['MODEL_PATH'], no_char_decoder=args['--no-char-decoder'])
    if args['--cuda']:
        model = model.to(torch.device("cuda:0"))
    return model

def read_corpus(lines, source):
    """ Read file, where each sentence is dilineated by a `\n`.
    @param file_path (str): path to file containing corpus
    @param source (str): "tgt" or "src" indicating whether text
        is of the source language or target language
    """
    data = []
    for line in lines:
        sent = nltk.word_tokenize(line)
        # only append <s> and </s> to the target sentence
        if source == 'tgt':
            sent = ['<s>'] + sent + ['</s>']
        data.append(sent)

    return data

def decode_stream(args, st, model):
    """ Performs decoding on a test set, and save the best-scoring decoding results.
    If the target gold-standard sentences are given, the function also computes
    corpus-level BLEU score.
    @param args (Dict): args from cmd line
    """
    test_data_src = read_corpus(args['TEST_SOURCE_FILE'], source='src')
    if args['TEST_TARGET_FILE']:
        test_data_tgt = read_corpus(args['TEST_TARGET_FILE'], source='tgt')

    hypotheses = run.beam_search(model, test_data_src,
                             beam_size=int(args['--beam-size']),
                             max_decoding_time_step=int(args['--max-decoding-time-step']))

    # if args['TEST_TARGET_FILE']:
    #     top_hypotheses = [hyps[0] for hyps in hypotheses]
    #     bleu_score = run.compute_corpus_level_bleu_score(test_data_tgt, top_hypotheses)
    #     st.success(f"Corpus BLEU: {bleu_score * 100}")

    
    for src_sent, hyps in zip(test_data_src, hypotheses):
        top_hyp = hyps[0]
        detokenizer = run.TreebankWordDetokenizer()
        detokenizer.DOUBLE_DASHES = (re.compile(r'--'), r'--')
        hyp_sent = detokenizer.detokenize(top_hyp.value)
        # hyp_sent = ' '.join(top_hyp.value)
        st.success(f"Model's Eng Translation : {hyp_sent}\n")


def stream_run():
    download_data()
    translator = google_translator()
    args = {}
    args["--cuda"] = False
    args["MODEL_PATH"] = "model.bin"
    args["--max-decoding-time-step"] = 70
    args["--beam-size"] = 5
    args["--seed"] = 0
    args["train"] = False
    args['--no-char-decoder'] = False
    args["decode"] = True

    assert(torch.__version__ >= "1.0.0"), "Please update your installation of PyTorch. You have {} and you should have version 1.0.0".format(torch.__version__)
    seed = int(args['--seed'])
    torch.manual_seed(seed)
    if args['--cuda']:
        torch.cuda.manual_seed(seed)
    np.random.seed(seed * 13 // 7)
    model = load_model(args)

    st.title("Neural Machine Translation")
    # st.title("Spanish To English Translation")
    html_temp = """
    <div style="background-color:brown;padding:10px">
    <h2 style="color:yellow;text-align:center;">Spanish to English Translation</h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)
    st.text("")
    st.text("")
    st.success("Welcome! Please enter a sentence!")   
    st.sidebar.markdown('''
    # What is this project?
    - This is a Neural Machine Translation Model, which translates from Spanish to English. :books:
    - If you know Spanish go ahead and enter the sentence you want to translate! :bulb:
    - If you are just like me, don't worry, enter the sentence in English and Google will convert it to Spanish before feeding it to the model. :brain:
    - You can compare how Google predicts the english translation and what our model generates. :eyes:
    # What is Beam Size?
    - It determines the number of alternatives our model looks at while predicting each word, to get the best combination of words for the translation.
    - Read more about it on this [blog](https://towardsdatascience.com/an-intuitive-explanation-of-beam-search-9b1d744e7a0f)
    ''')
    
    st.sidebar.markdown('''Liked it? Give a :star:  on GitHub [here](https://github.com/sahilkhose/Neural_Machine_Translation)''')

    lang = st.radio(
    "Input Language",
    ('English', 'Spanish'))

    en_sentence = ""
    es_sentence = ""

    if lang == "Spanish":
        es_sentence = st.text_input("Spanish Sentence")
    else:
        en_sentence = st.text_input("English Sentence", help="Enter sentence to translate")

    # beam_size = st.text_input("Beam Size", "5")
    beam_size = st.slider(label="Beam Size (recommended: 5)", min_value=1, max_value=20, value=5)
    if st.button("Translate"):  
        if es_sentence != "" or en_sentence != "":
            if beam_size != "":
                args["--beam-size"] = int(beam_size)
            if lang != "Spanish":
                es_sentence = translator.translate(en_sentence, lang_src="en", lang_tgt="es")
            en_google = translator.translate(es_sentence, lang_tgt="en")
            args["TEST_TARGET_FILE"] = [en_google]
            args["TEST_SOURCE_FILE"] = [es_sentence]
            st.success(f"Spanish sentence           : {es_sentence}")
            st.success(f"Google's Eng Translation   : {en_google}")
            decode_stream(args, st, model)
        else:
            st.exception(RuntimeError("Enter a sentence to translate!"))



if __name__ == "__main__":
    st.set_page_config(
    initial_sidebar_state="expanded",
    page_title="Machine Translation"
    )
    stream_run()