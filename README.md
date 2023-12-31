# LRCS
# # 1 Overview
Context-based Transfer Learning for Low Resource Code Summarization
Source code summaries improve the readability and intelligibility of code, help developers understand programs, andimprove the efficiency of software maintenance and upgrade processes. Unfortunately, these code comments are oftenmismatched, missing, or outdated in software projects, resulting in developers needing to infer functionality from source code, affecting the efficiency of software maintenance and evolution. Various methods based on neuronal networks are proposed to solve the problem of synthesis of source code. However, the current work is being carried out on resource-rich programming languages such as Java and Python, and some low-resource languages may not perform well. In order to solve the above challenges, we pro-
pose a context-based transfer learning model for Low Resource Code Summarization (LRCS), which learns the common information fromthe language with rich resources, and then transfers it to the target language model for further learning. It consists of two components: the summary generation component is used to learn the syntactic and semantic information of the code, and the learning transfer component is used to improve the generalization ability of
the model in the learning process of cross-language code summarization. Experimental results show that LRCS outperforms baseline methods in code summarization in terms of sentence-level BLEU, corpus-level BLEU and METEOR.For example, LRCS improves corpu-level BLEU scores by 52.90%, 41.10% and 14.97%, respectively, compared to baseline methods.
# # 2 Project Structure
- `data`
    - `*big_data:Save the code_search_nets dataset`
    - `*go:Save the processed data of the corresponding Go type generated by running pre_process.py.  Please create the remaining two folders in the same way as this folder.`
    - `*datasets:Save the folder at the project's root path, for example, 'go' (put the entire folder directly into it)`
- `models:Save the configuration files needed at runtime. When you need experimental results for the corresponding language, you can change the corresponding YAML file.`
- `scripts:Save the shell script used to run the project.`

# # 3 How it operator
-`Firstly,we pre-train on the big_data by running pre_process.py and generate three files in the corresponding root directory folder.`    
-`Secondly,we will place the generated files and folders into the 'datasets' folder.`
-`Lastly,we adjust the contents of the YAML files in the 'models' directory and run the 'train.sh' script in the 'scripts' directory to complete the training, or the 'test.sh' script to perform testing.`

# # 4 Environment configuration
- `ubuntu`
- `tensorflow>=2.0`
- `python>=3.6` 
  
      
