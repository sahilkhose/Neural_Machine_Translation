#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2019-20: Homework 5
run.py: Run Script for Simple NMT Model
Pencheng Yin <pcyin@cs.cmu.edu>
Sahil Chopra <schopra8@stanford.edu>
Kuangcong Liu <cecilia4@stanford.edu>

Usage:
    translate.py train --train-src=<file> --train-tgt=<file> --dev-src=<file> --dev-tgt=<file> --vocab=<file> [options]
    translate.py decode [options] MODEL_PATH TEST_SOURCE_FILE OUTPUT_FILE
    translate.py decode [options] MODEL_PATH TEST_SOURCE_FILE TEST_TARGET_FILE OUTPUT_FILE

Options:
    -h --help                               show this screen.
    --cuda                                  use GPU
    --train-src=<file>                      train source file
    --train-tgt=<file>                      train target file
    --dev-src=<file>                        dev source file
    --dev-tgt=<file>                        dev target file
    --vocab=<file>                          vocab file
    --seed=<int>                            seed [default: 0]
    --batch-size=<int>                      batch size [default: 32]
    --word-embed-size=<int>                 word embedding size [default: 256]
    --hidden-size=<int>                     hidden size [default: 256]
    --clip-grad=<float>                     gradient clipping [default: 5.0]
    --log-every=<int>                       log every [default: 10]
    --max-epoch=<int>                       max epoch [default: 30]
    --input-feed                            use input feeding
    --patience=<int>                        wait for how many iterations to decay learning rate [default: 5]
    --max-num-trial=<int>                   terminate training after how many trials [default: 5]
    --lr-decay=<float>                      learning rate decay [default: 0.5]
    --beam-size=<int>                       beam size [default: 5]
    --sample-size=<int>                     sample size [default: 5]
    --lr=<float>                            learning rate [default: 0.001]
    --uniform-init=<float>                  uniformly initialize all parameters [default: 0.1]
    --save-to=<file>                        model save path [default: model.bin]
    --valid-niter=<int>                     perform validation after how many iterations [default: 2000]
    --dropout=<float>                       dropout [default: 0.3]
    --max-decoding-time-step=<int>          maximum number of decoding time steps [default: 70]
    --no-char-decoder                       do not use the character decoder
"""
from google_trans_new import google_translator
from docopt import docopt
import run as runs
import numpy as np 
import sys
import torch
import re 

def decode(args: runs.Dict[str, str]):
    """ Performs decoding on a test set, and save the best-scoring decoding results.
    If the target gold-standard sentences are given, the function also computes
    corpus-level BLEU score.
    @param args (Dict): args from cmd line
    """

    # print("load test source sentences from [{}]".format(args['TEST_SOURCE_FILE']), file=sys.stderr)
    test_data_src = runs.read_corpus(args['TEST_SOURCE_FILE'], source='src')
    if args['TEST_TARGET_FILE']:
        # print("load test target sentences from [{}]".format(args['TEST_TARGET_FILE']), file=sys.stderr)
        test_data_tgt = runs.read_corpus(args['TEST_TARGET_FILE'], source='tgt')

    # print("load model from {}".format(args['MODEL_PATH']), file=sys.stderr)
    model = runs.NMT.load(args['MODEL_PATH'], no_char_decoder=args['--no-char-decoder'])

    if args['--cuda']:
        model = model.to(torch.device("cuda:0"))

    hypotheses = runs.beam_search(model, test_data_src,
                             beam_size=int(args['--beam-size']),
                             max_decoding_time_step=int(args['--max-decoding-time-step']))

    if args['TEST_TARGET_FILE']:
        top_hypotheses = [hyps[0] for hyps in hypotheses]
        bleu_score = runs.compute_corpus_level_bleu_score(test_data_tgt, top_hypotheses)
        print("__"*80)
        print('Corpus BLEU: {}'.format(bleu_score * 100), file=sys.stderr)

    with open(args['OUTPUT_FILE'], 'w') as f:
        for src_sent, hyps in zip(test_data_src, hypotheses):
            top_hyp = hyps[0]
            detokenizer = runs.TreebankWordDetokenizer()
            detokenizer.DOUBLE_DASHES = (re.compile(r'--'), r'--')
            hyp_sent = detokenizer.detokenize(top_hyp.value)
            # hyp_sent = ' '.join(top_hyp.value)
            f.write(hyp_sent + '\n')

def write_doc(sent, file):
	with open(file, mode="w") as f:
		f.write(sent)

if __name__ == "__main__":

	args = docopt(__doc__)
	print("hi")

	translator = google_translator()
	en_sentence = str(input("Enter the sentence in English: \n"))
	write_doc(en_sentence, args['TEST_TARGET_FILE'])
	es_sentence = translator.translate(en_sentence, lang_src="en", lang_tgt="es")
	write_doc(es_sentence, args['TEST_SOURCE_FILE'])
	en_google = translator.translate(es_sentence, lang_tgt="en")


	assert(torch.__version__ >= "1.0.0"), "Please update your installation of PyTorch. You have {} and you should have version 1.0.0".format(torch.__version__)
	# seed the random number generators
	seed = int(args['--seed'])
	torch.manual_seed(seed)
	if args['--cuda']:
	    torch.cuda.manual_seed(seed)
	np.random.seed(seed * 13 // 7)

	if args['train']:
	    train(args)
	elif args['decode']:
	    decode(args)
	else:
	    raise RuntimeError('invalid run mode')


	with open(args['OUTPUT_FILE'], mode="r") as f:
		en_model = f.read()
	print(f"en_sentence: {en_sentence}")
	print(f"es_sentence: {es_sentence}")
	print(f"en_google: {en_google}")
	print(f"en_model: {en_model}")