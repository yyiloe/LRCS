import sys
from nltk.translate.meteor_score import meteor_score
import os

root_path=os.path.dirname(os.path.dirname(__file__))

def nltk_sentence_meteor(hypothesis, reference):
    score = round(meteor_score([reference], hypothesis), 4)
    return score


if __name__ == '__main__':
    scores = 0
    cnt = 0

    # hyp_path = os.path.join(root_path,"models/ruby/java/eval/predictions.txt.1900000")
    ref_path = os.path.join(root_path,"datasets/ruby/eval/tgt_title.txt")
    # ref_path = os.path.join(root_path,"datasets/ruby/test/tgt_title.txt")
    hyp_path = os.path.join(root_path,"models_old/ruby/fsl_new/eval/predictions.txt.2300000")

    # hyp_path = os.path.join(root_path,"models/ruby/base_model/eval/predictions.txt.2100000")
    # hyp_path = os.path.join(root_path,"models/ruby/python/eval/predictions.txt.2100000")
    # hyp_path = os.path.join(root_path,"models/java/run/eval/predictions.txt.3800000")
    # ref_path = os.path.join(root_path,"datasets/ruby/test/tgt_title.txt")
    with open(hyp_path, "r", encoding="utf-8") as hyp_file:
        with open(ref_path,"r", encoding="utf-8") as ref_file:
            for hyp, ref in zip(hyp_file.readlines(), ref_file.readlines()):
                # if (hyp=='\n'):
                #     continue
                hyp=hyp.split()
                ref=ref.split()
                # if(len(hyp)<4 or len(hyp)>(50)):
                #     continue
                score = nltk_sentence_meteor(hyp, ref)

                # try:
                #     score = nltk_sentence_bleu(hyp, ref)
                # except ValueError:
                #     continue
                # except ZeroDivisionError:
                #     continue
                # if (score<=0.01):
                #     continue
                scores += score
                cnt += 1
                # if(cnt%100):
                #     print(cnt,score,hyp,ref)
                

    print("METEOR =", round(scores/cnt*100,2))
    print(cnt)
