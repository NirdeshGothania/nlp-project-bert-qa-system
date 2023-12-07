# Question Answering System with BERT

## Overview
This project utilizes the BERT (Bidirectional Encoder Representations from Transformers) model for question answering. BERT is a powerful pre-trained language model that can be fine-tuned for specific tasks. In this case, the model is fine-tuned to answer questions based on a given context.

# Project Structure
The project is organized into several Python scripts, each responsible for a specific part of the ML pipeline. The main script orchestrates the entire process.

- `main.py`: The main script containing the code for loading the pre-trained BERT model, preparing the question-answering dataset, fine-tuning the model, and providing an interactive interface for users to ask questions.

- `kangaroo.txt`: A sample text file containing information about kangaroos. This file serves as the context for the question-answering task.

- `requirements.txt`: A file listing the required Python packages for running the project.

## Execution
First, make sure that the current directory opened in the terminal is the correct directory in which all the python script files are present alongwith dataset file in .txt format.

1. And make sure you have the required dependencies installed before running the code. You can install them by executing the following command in your terminal:

```
pip install -r requirements.txt
```

This command installs the necessary Python packages listed in the requirements.txt file, including:
- torch
- transformers
- tqdm

2. How to run the code; Use this command on the terminal:  
```
python main.py
```

3. **Input Format**: Once the model is fine-tuned, you can interactively ask questions about kangaroos using the provided context. The script will tokenize your question, pass it through the BERT model, and provide an answer.
```
Type your question: How fast can kangaroos hop?
```

4. **Output Information**:

When interacting with the model using the main.py script, you can expect the following output information:

- **Fine-Tuning**:
During the fine-tuning process, the script will display progress information for each epoch. You will see the average training loss for each epoch.
```
Epoch 1/10, Avg Train Loss: 6.234
Epoch 2/10, Avg Train Loss: 6.178
...
```

- **User Interaction**:
After typing your question, the model will generate an answer based on the provided context. The answer will be displayed in the console.
```
Answer: Kangaroos exhibit an iconic hopping motion, covering large distances with impressive leaps.
```