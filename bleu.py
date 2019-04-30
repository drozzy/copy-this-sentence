from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

chencherry = SmoothingFunction()

def bleu(reference, hypothesis, eos_token): 
	
	ref_i = reference.index(eos_token) if eos_token in reference else -1
	if ref_i >= 0:
		reference = reference[0: ref_i + 1]

	h_i = hypothesis.index(eos_token) if eos_token in hypothesis else -1
	if h_i >= 0:
		hypothesis = hypothesis[0: h_i + 1]

	return sentence_bleu([reference], hypothesis, smoothing_function=chencherry.method3)

def bleu_ignore_eos(reference, hypothesis, eos_token): 
    if eos_token in reference:
        reference = reference[0: reference.index(eos_token)]

    if eos_token in hypothesis:
        hypothesis = hypothesis[0: hypothesis.index(eos_token)]
        
    return sentence_bleu([reference], hypothesis, smoothing_function=chencherry.method3)