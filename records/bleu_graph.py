import sys
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import os
from matplotlib import pyplot as plt 
from tqdm import tqdm

root_path=os.path.dirname(os.path.dirname(__file__))

def nltk_sentence_bleu(hypothesis, reference):
    smooth = SmoothingFunction()
    score = sentence_bleu([reference], hypothesis,weights=(0.25,0.25,0.25,0.25) ,smoothing_function=smooth.method4)
    return score


if __name__ == '__main__':
    plt.title("FSL java") 
    plt.xlabel("steps/10000") 
    plt.ylabel("BLEU") 
    if len(sys.argv) > 1:
        lang = sys.argv[1]
    else:
        lang = "python"
    ratio=["10%","20%","30%","40%","50%"]

    for r in tqdm(ratio):
        bleu=[]
        steps=[]
        for i in range(100000,1100000,100000):
            scores = 0
            cnt = 0

            hyp_path = os.path.join(root_path,"models/ast_fsl/"+r+"/eval/predictions.txt."+str(i))
            ref_path = os.path.join(root_path,"datasets/java_v2/eval/tgt_title.csv")
            with open(hyp_path, "r", encoding="utf-8") as hyp_file:
                with open(ref_path,"r", encoding="utf-8") as ref_file:
                    for hyp, ref in zip(hyp_file.readlines(), ref_file.readlines()):
                        if (hyp=='\n'):
                            continue
                        hyp=hyp.split()
                        ref=ref.split()
                        try:
                            score = nltk_sentence_bleu(hyp, ref)
                        except ValueError:
                            continue
                        except ZeroDivisionError:
                            continue
                        scores += score
                        cnt += 1
            steps.append(i/10000)
            bleu.append(round(scores/cnt*100,2))

        plt.plot(steps,bleu) 
    
    bleu=[]
    steps=[]
    for i in range(100000,1100000,100000):
        scores = 0
        cnt = 0

        hyp_path = os.path.join(root_path,"models/trans_java/run/eval/predictions.txt."+str(i))
        ref_path = os.path.join(root_path,"datasets/java_v2/eval/tgt_title.csv")
        with open(hyp_path, "r", encoding="utf-8") as hyp_file:
            with open(ref_path,"r", encoding="utf-8") as ref_file:
                for hyp, ref in zip(hyp_file.readlines(), ref_file.readlines()):
                    if (hyp=='\n'):
                        continue
                    hyp=hyp.split()
                    ref=ref.split()
                    try:
                        score = nltk_sentence_bleu(hyp, ref)
                    except ValueError:
                        continue
                    except ZeroDivisionError:
                        continue
                    scores += score
                    cnt += 1
        steps.append(i/10000)
        bleu.append(round(scores/cnt*100,2))

    plt.plot(steps,bleu) 

    plt.legend(["10%","20%","30%","40%","50%","trans"])
    plt.savefig(os.path.join(root_path,'records/FSL Java.png'))
            