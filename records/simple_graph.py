import sys
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
import os
from matplotlib import pyplot as plt
from tqdm import tqdm

root_path=os.path.dirname(os.path.dirname(__file__))

def nltk_sentence_bleu(hypothesis, reference):
    smooth = SmoothingFunction()
    score = sentence_bleu([reference], hypothesis,weights=(0.25,0.25,0.25,0.25) ,smoothing_function=smooth.method2)
    # score = sentence_bleu([reference], hypothesis,weights=(0,0,0,1) ,smoothing_function=smooth.method2)
    return score

def nltk_sentence_meteor(hypothesis, reference):
    score = round(meteor_score([reference], hypothesis), 4)
    return score

if __name__ == '__main__':
    plt.title("FSL ruby")
    plt.xlabel("steps/10000")
    plt.ylabel("BLEU")
    if len(sys.argv) > 1:
        lang = sys.argv[1]
    else:
        lang = "python"

    bleu=[]
    steps=[]
    for i in tqdm(range(100000,2000000,100000)):
        scores = 0
        cnt = 0
        # hyp_path = os.path.join(root_path,"models/emb64/eval/predictions.txt."+str(i))
        # hyp_path = os.path.join(root_path,"models/ruby/python/eval/predictions.txt."+str(i))
        # hyp_path = os.path.join(root_path,"models/emb64/eval/predictions.txt."+str(i))
        # hyp_path = os.path.join(root_path,"models/java/run/eval/predictions.txt."+str(i))
        # hyp_path = os.path.join(root_path,"models/ruby/fsl/eval/predictions.txt."+str(i))
        hyp_path = os.path.join(root_path,"exps/java/eval/predictions.txt."+str(i))
        # ref_path = os.path.join(root_path,"datasets/ruby/eval/tgt_title.txt")
        ref_path = os.path.join(root_path,"datasets/java/eval/tgt_title.txt")
        with open(hyp_path, "r", encoding="utf-8") as hyp_file:
            with open(ref_path,"r", encoding="utf-8") as ref_file:
                for hyp, ref in zip(hyp_file.readlines(), ref_file.readlines()):
                    if (hyp=='\n'):
                        continue
                    # hyp=hyp.split()
                    # ref=ref.split()
                    try:
                        score = nltk_sentence_meteor(hyp, ref)
                    except ValueError:
                        continue
                    except ZeroDivisionError:
                        continue
                    scores += score
                    cnt += 1
        steps.append(i/10000)
        bleu.append(round(scores/cnt*100,2))

    plt.plot(steps,bleu)


    plt.savefig(os.path.join(root_path,'records/java.png'))
