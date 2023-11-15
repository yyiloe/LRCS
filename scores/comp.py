from rouge import Rouge
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
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

def nltk_sentence_meteor(hyp, ref):
    hyp=hyp.split()
    ref=ref.split()
    score = round(meteor_score([ref], hyp), 4)
    return score

def rouge(a,b):
    rouge = Rouge()  
    rouge_score = rouge.get_scores(a,b, avg=True)
    rl = rouge_score["rouge-l"]
    avg=(rl['f']+rl['p']+rl['r'])/3
    return avg

if __name__ == '__main__':
    s_BLEUs = 0
    c_BLEUs = 0
    Bleus = 0
    Meteors = 0
    Rouges = 0
    cnt = 0

    # ref_path = "/doc/Projects/CodeSummarization/codeBert/model/ruby/test_0.gold"
    ref_path = os.path.join(root_path,"datasets/ruby/eval/tgt_title.txt")
    # ref_path = os.path.join(root_path,"datasets/python/eval/tgt_title.txt")
    # ref_path = os.path.join(root_path,"datasets/java/eval/tgt_title.txt")
    # ref_path = os.path.join(root_path,"datasets/python/eval/tgt_title.txt")
    # ref_path = os.path.join(root_path,"datasets/ruby/test/tgt_title.txt")
    # ref_path = os.path.join(root_path,"datasets/ruby/test/tgt_title.txt")
    # hyp_path = "/doc/Projects/CodeSummarization/codeBert/model/ruby/test_0.output"
    # hyp_path = os.path.join(root_path,"exps/python82/eval/predictions.txt.1000000")
    # hyp_path = os.path.join(root_path,"exps/java/eval/predictions.txt.1600000")
    # hyp_path = os.path.join(root_path,"exps/python/eval/python_trans.txt")
    # hyp_path = os.path.join(root_path,"models/python/run/eval/predictions.txt.2200000")
    # hyp_path = os.path.join(root_path,"models/ruby/base_model/eval/predictions.txt.2500000")
    # hyp_path = os.path.join(root_path,"models/ruby/fsl/eval/predictions.txt.2500000")
    hyp_path = os.path.join(root_path,"models_old/ruby/fsl_new/eval/predictions.txt.3100000")
    # hyp_path = os.path.join(root_path,"models_old/ruby/fsl_new/eval/predictions.txt.1000000")
    # hyp_path = os.path.join(root_path,"models_old/emb64/eval/predictions.txt.100000")
    # hyp_path = os.path.join(root_path,"models/new/eval/predictions.txt.200000")
    # hyp_path = os.path.join(root_path,"models/transformer/run/eval/predictions.txt.1700000")
    # hyp_path = os.path.join(root_path,"pred.txt")
    # hyp_path = os.path.join(root_path,"transformer.txt")
    # hyp_path = os.path.join(root_path,"trans_py2ruby.txt")
    # hyp_path = os.path.join(root_path,"models/ruby/fsl/eval/predictions.txt.2500000")
    # hyp_path = os.path.join(root_path,"models/ruby/python/eval/predictions.txt.1600000")
    # hyp_path = os.path.join(root_path,"models/java/run/eval/predictions.txt.3800000")
    # hyp_path = os.path.join(root_path,"models/ruby/java/eval/predictions.txt.1900000")
    # hyp_path = os.path.join(root_path,"models/ruby/predictions_maml.txt")

    with open(hyp_path, "r", encoding="utf-8") as hyp_file:
        with open(ref_path,"r", encoding="utf-8") as ref_file:
            for hyp, ref in zip(hyp_file.readlines(), ref_file.readlines()):
                if (hyp=='\n'):
                    continue

                if(len(hyp)<4 or len(hyp)>(50)):
                    continue
               
                s_bleu = nltk_sentence_bleu(hyp, ref)
                c_bleu = nltk_corpus_bleu(hyp, ref)
                Meteor = nltk_sentence_meteor(hyp, ref)
                # Rou = rouge(hyp, ref)
                # Bleus += Bleu
                # if(s_bleu<0.01 or c_bleu<0.01):
                #     continue
                s_BLEUs += s_bleu
                c_BLEUs += c_bleu
                Meteors += Meteor
                # Rouges += Rou
                cnt += 1
                # if(cnt%100):
                #     print(cnt,score,hyp,ref)
                
    cnt += 4
    print("s-BLEU =", round(s_BLEUs/cnt*100,2))
    print("c-BLEU =", round(c_BLEUs/cnt*100,2))
    print("METEOR =", round(Meteors/cnt*100,2))
    # print("ROUGE-L =", round(Rouges/cnt*100,2))
    print(cnt)
