# Personalizing Conversational AI Using the Big Five Personality Traits

## Practical implementation of the Master's Thesis submitted in partial fulfillment to the requirements for a Master's Degree in Computational Linguistics at Ruhr-University Bochum

### Abstract
As human-machine interactions become ever more frequent, the trend in Artificial Intelligence goes 
towards personalized conversational agents. These agents mimic human behavior in respect to 
personality and are either bestowed with their own persona or have received the ability to read and 
respond to human emotion. Studies attest the success of such agents with increased measurements in 
language competence and user satisfaction.

This thesis takes an approach to personalization that is based in psycholinguistic and personality 
theory. The aim is to further improve the user experience in human-chatbot interaction by developing
a conversational agent that is flexible in its linguistic behavior and adapts to the user's 
personality. It is assumed, that the user will feel more comfortable talking to the conversational 
agent, when it embodies a personality that is similar to the user's, as to when it is dissimilar. 
To that end, a new dataset comprising personality annotated dialogues was collected. Furthermore, 
state-of-the-art Transformer neural networks and methods such as transfer learning are used for the 
implementation of the agent. 

In this context, the thesis covers fundamental linguistic dialogue theory, as well as pioneering and 
contemporary techniques in dialogue response generation and machine learning. The focus will lie on 
Transformer-based conversational agents and end-to-end approaches. Unfortunately, the thesis is not 
able to provide a clear conclusion on the benefits of incorporating personality traits in 
personalization, yet offers some discussion on issues with the approach in general and possible 
improvements in the future. 

### About
The practical part of the thesis is divided in two parts. The first part is concerned with data understanding 
and an analysis of the collected dataset. This part includes automatic and manual processes, i.e. cleaning of the data, 
analysis of the accompanying personality questionnaires and a calculation of speaker alignment. The second part deals with 
machine learning engineering and training a language model for conditioned dialogue response generation.

### Requirements
The project was implemented in Python3.7 and uses external libraries that can be installed via the requirements.txt.
It is advised to create a virtual environment for the project.


### Steps

#### Step 1.1: Clean and assess the chat data
To automatically clean the chat data and e.g. remove special symbols and chronologically sort the data, 
execute file **data_understanding/process_chats.py**.
This will also provide a first statistical analysis of the chats in terms of number of chats, number of speakers 
and length of messages. It is advised to conduct manual cleaning of the data, as well.

#### Step 1.2: Assess personality questionnaires
The dataset comes with raw personality scores collected via the **BFI-S questionnaire**. To calculate the final scores 
and remove scores of subjects that did not partake in the chats, execute file **calculate_personality_scores.py**.
The program will also provide a visualization of the final scores, add personality labels to the chat data and split the 
data into individual chats for further processing.

#### Step 1.3: Calculate Alignment and Linguistic Style Matching
Using the files resulting from the previous step, each chat is analyzed using the **LIWC** tool (*http://liwc.wpengine.com/*).
Of interest is the use of function category words (e.g. negation, personal pronouns and auxiliary verbs)
per user. The liwc scores are provided in **outputs/liwc/** and can be analyzed using **calculate_liwc_results.py**.
This will provide scores for the dataset overall, as well as a comparison of scores for chats between only extroverted 
and mixed personality pairs.

#### Step 2.1: Prepare the dataset
Using the annotated corpus resulting from **Step 1.2**, add context and distractor phrases to the dialogue 
data. Executing **add_context_columns.py** will output a modified csv file, that contains the reformatted the data and
can be used for training the dialogue model.

#### Step 2.2: Preprocess the dataset
For preprocessing, the dataset is tokenized and transformed into a Tensor Dataset. The dataset contains
label for language modeling and next-sentence prediction.

#### Step 2.3: Fine-tune the model on a multi-task objective
The dialogue model is fine-tuned on a multi-task objective using the preprocessed dataset. **train.py** 
will preprocess the data and begin fine-tuning. Model checkpoints are saved at modeling/runs/. The pretrained base model
is loaded from the **HuggingFace Transformers community model hub**. 

### Notes:
As of now, the implementation produces some issues and is not running bug-free. The main issue is an
out-of-bounds error occurring during fine-tuning. 

