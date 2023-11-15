import sys
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu, SmoothingFunction
import os

root_path=os.path.dirname(os.path.dirname(__file__))

def nltk_sentence_bleu(hyp, ref):
    hyp=hyp.split()
    ref=ref.split()
    smooth = SmoothingFunction()
    score = sentence_bleu([ref], hyp,weights=(0.25,0.25,0.25,0.25) ,smoothing_function=smooth.method2)
    return score
def nltk_corpus_bleu(hyp, ref):
    smooth = SmoothingFunction()
    hyp=hyp.split()
    ref=ref.split()
    score = corpus_bleu([ref], [hyp], weights=(0.25,0.25,0.25,0.25) ,smoothing_function=smooth.method2)
    return score
# def nltk_sentence_bleu1(hypothesis, reference):
#     smooth = SmoothingFunction()
#     score = sentence_bleu([reference], hypothesis,weights=(1,0,0,0) ,smoothing_function=smooth.method2)
#     return score

# def nltk_sentence_bleu2(hypothesis, reference):
#     smooth = SmoothingFunction()
#     score = sentence_bleu([reference], hypothesis,weights=(0.5,0.5,0,0) ,smoothing_function=smooth.method2)
#     return score

# def nltk_sentence_bleu3(hypothesis, reference):
#     smooth = SmoothingFunction()
#     score = sentence_bleu([reference], hypothesis,weights=(0.33,0.33,0.33,0) ,smoothing_function=smooth.method2)
#     return score

# def nltk_sentence_bleu4(hypothesis, reference):
#     smooth = SmoothingFunction()
#     score = sentence_bleu([reference], hypothesis,weights=(0.25,0.25,0.25,0.25) ,smoothing_function=smooth.method2)
#     return score


if __name__ == '__main__':
    s_BLEUs = 0
    c_BLEUs = 0
    scores = 0
    scores1 = 0
    scores2 = 0
    scores3 = 0
    scores4 = 0
    cnt = 0
    if len(sys.argv) > 1:
        lang = sys.argv[1]
    else:
        lang = "python"


    # hyp_path = os.path.join(root_path,"models/ruby/java/eval/predictions.txt.1900000")
    ref_path = os.path.join(root_path,"datasets/ruby/eval/tgt_title.txt")
    # ref_path = os.path.join(root_path,"datasets/ruby/test/tgt_title.txt")
    # hyp_path = os.path.join(root_path,"models/ruby/predictions_maml.txt")
    # hyp_path = os.path.join(root_path,"models/ruby/fsl_new/eval/predictions.txt.900000")
    # hyp_path = os.path.join(root_path,"models/ruby/fsl_new/eval/predictions.txt.1600000")
    hyp_path = os.path.join(root_path,"models_old/ruby/fsl_new/eval/predictions.txt.2300000")
    # hyp_path = os.path.join(root_path,"models/ruby/fsl/eval/predictions.txt.2500000")
    # hyp_path = os.path.join(root_path,"transformer.txt")
    # hyp_path = os.path.join(root_path,"trans_py2ruby.txt")
    # hyp_path = os.path.join(root_path,"pj2ruby.txt")
    # hyp_path = os.path.join(root_path,"models/ruby/base_model/eval/predictions.txt.2100000")
    # hyp_path = os.path.join(root_path,"models/ruby/python/eval/predictions.txt.2100000")
    # hyp_path = os.path.join(root_path,"models/java/run/eval/predictions.txt.3800000")
    # ref_path = os.path.join(root_path,"datasets/ruby/test/tgt_title.txt")
    with open(hyp_path, "r", encoding="utf-8") as hyp_file:
        with open(ref_path,"r", encoding="utf-8") as ref_file:
            for hyp, ref in zip(hyp_file.readlines(), ref_file.readlines()):

                s_bleu = nltk_sentence_bleu(hyp, ref)
                c_bleu = nltk_corpus_bleu(hyp, ref)
                # if(s_bleu<0.01 or c_bleu<0.01):
                #     continue
                # score1 = nltk_sentence_bleu1(hyp, ref)
                # score2 = nltk_sentence_bleu2(hyp, ref)
                # score3 = nltk_sentence_bleu3(hyp, ref)
                # score4 = nltk_sentence_bleu4(hyp, ref)
                s_BLEUs += s_bleu
                c_BLEUs += c_bleu
                # scores += score
                # scores1 += score1
                # scores2 += score2
                # scores3 += score3
                # scores4 += score4
                
                
                
                cnt += 1
                # if(cnt%100):
                #     print(cnt,score,hyp,ref)


                

    print("s-BLEU =", round(s_BLEUs/cnt*100,2))
    print("c-BLEU =", round(c_BLEUs/cnt*100,2))
    # print("BLEU1 =", round(scores1/cnt*100,2))
    # print("BLEU2 =", round(scores2/cnt*100,2))
    # print("BLEU3 =", round(scores3/cnt*100,2))
    # print("BLEU4 =", round(scores4/cnt*100,2))
    print(cnt)
