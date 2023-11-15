import os

root_path=os.path.dirname(os.path.dirname(__file__))
tgt_path = os.path.join(root_path,"models/ruby/fsl/eval/predictions.txt.4000000")
out_path = os.path.join(root_path,"test_comp/prediction.txt")

with open(tgt_path, "r", encoding="utf-8") as tgt_file:
    with open(out_path,"w+", encoding="utf-8") as ref_file:
        for i,hyp in enumerate(tgt_file.readlines()):
            ref_file.write(str(i)+'	'+hyp)