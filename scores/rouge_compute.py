from rouge import Rouge
from rouge_metric import PyRouge
import os
from tqdm import tqdm

root_path=os.path.dirname(os.path.dirname(__file__))
rouge = PyRouge(rouge_l=True, rouge_w=True,
                rouge_w_weight=1.2, rouge_s=True, rouge_su=True, skip_gap=4)

def rouge_origin(a,b):
    rouge = Rouge()  
    rouge_score = rouge.get_scores(a,b, avg=True) # a和b里面包含多个句子的时候用
    rouge_score1 = rouge.get_scores(a,b)  # a和b里面只包含一个句子的时候用
    rl = rouge_score["rouge-1"]
    # avg=(rl['f']+rl['p']+rl['r'])/3
    # return avg
    return rl['f']

def rouge_1(a,b):
    scores = rouge.evaluate_tokenized([a], [b])
    rl=scores["rouge-l"]
    avg=(rl['f']+rl['p']+rl['r'])/3
    return avg

if __name__ == '__main__':
    scores = 0
    cnt = 0

    # hyp_path = os.path.join(root_path,"models/ruby/java/eval/predictions.txt.1900000")
    ref_path = os.path.join(root_path,"datasets/ruby/eval/tgt_title.txt")
    # hyp_path = os.path.join(root_path,"models/ruby/fsl_new/eval/predictions.txt.1600000")
    # ref_path = os.path.join(root_path,"datasets/ruby/test/tgt_title.txt")
    # hyp_path = os.path.join(root_path,"models/ruby/predictions_maml.txt")
    # hyp_path = os.path.join(root_path,"models/ruby/fsl_new/eval/predictions.txt.900000")
    # hyp_path = os.path.join(root_path,"models/ruby/base_model/eval/predictions.txt.3400000")
    # hyp_path = os.path.join(root_path,"models/ruby/python/eval/predictions.txt.2100000")
    # hyp_path = os.path.join(root_path,"models/java/run/eval/predictions.txt.3800000")
    hyp_path = os.path.join(root_path,"trans_py2ruby.txt")
    # hyp_path = os.path.join(root_path,"models/ruby/fsl/eval/predictions.txt.2500000")
    # hyp_path = os.path.join(root_path,"transformer.txt")
    
    

    
    
    with open(hyp_path, "r", encoding="utf-8") as hyp_file:
        with open(ref_path,"r", encoding="utf-8") as ref_file:
            for hyp, ref in tqdm(zip(hyp_file.readlines(), ref_file.readlines())):
                # score = rouge_origin(hyp, ref)
                # scores += score
                scores = rouge.evaluate_tokenized([hyp], [ref])
                print(scores)
                cnt += 1
                break

    # print("ROUGE-L =", round(scores/cnt*100,2))
    print(cnt)
