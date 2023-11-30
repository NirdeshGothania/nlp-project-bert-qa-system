import torch
from transformers import BertTokenizer, BertForQuestionAnswering, AdamW
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# load the pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased') # it tokenizes the words to numerical token IDs # Add CLS(begining, represn entire seq.) and SEP(separate seq., also at end)
model = BertForQuestionAnswering.from_pretrained('bert-base-uncased', config='bert-base-uncased') # These configuration files include information such as the number of layers, hidden size, attention heads, etc.

# it read text from a file
def read_text_from_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file: 
        text = file.read()
    return text


context = read_text_from_file('kangaroo.txt')

# QA dataset
qa_dataset = [
    {
        'context': "Kangaroos are marsupials that are native to Australia. They are known for their powerful hind legs, strong tails, and distinctive hopping motion. There are several species of kangaroos, including the red kangaroo, eastern grey kangaroo, and western grey kangaroo. Kangaroos are herbivores and mainly eat grasses and other vegetation. They have a unique reproductive system where females have a pouch in which they carry and nurse their young, called joeys.",
        'question': "What do kangaroos eat?",
        'answer': "Kangaroos are herbivores and mainly eat grasses and other vegetation."
    },
    {
        'context': "Kangaroos are fascinating creatures with incredible adaptations. They have powerful hind legs that allow them to cover large distances with impressive leaps. Kangaroos are social animals and often form groups known as mobs. The red kangaroo, the largest marsupial, is known for its distinctive reddish-brown fur and can reach speeds of up to 56 kilometers per hour.",
        'question': "How fast can a red kangaroo run?",
        'answer': "The red kangaroo can reach speeds of up to 56 kilometers per hour."
    },
    {
        'context': "The life cycle of a kangaroo is truly unique. Female kangaroos give birth to relatively undeveloped young, known as joeys. The tiny joeys, often no larger than a lima bean, continue their development in the mother's pouch. As they grow, joeys gradually spend more time outside the pouch but continue to nurse for an extended period.",
        'question': "How do kangaroos raise their young?",
        'answer': "Female kangaroos carry and nurse their young, called joeys, in their pouch. Joeys continue to nurse and develop outside the pouch as they grow."
    },
    {
        'context': "Kangaroos play a crucial role in the ecosystem by controlling vegetation through their grazing habits. Their unique digestive system allows them to efficiently extract nutrients from tough grasses. Kangaroos are also important culturally, often appearing in Aboriginal Australian stories and art as symbols of strength and adaptability.",
        'question': "What role do kangaroos play in the ecosystem?",
        'answer': "Kangaroos play a crucial role in the ecosystem by controlling vegetation through their grazing habits."
    },
    {
        'context': "The conservation of kangaroos is of great importance. While they are iconic symbols of Australia, some species face threats such as habitat loss and conflicts with human activities. Conservation efforts focus on preserving their natural habitats and addressing challenges to ensure the continued survival of these unique marsupials.",
        'question': "Why is the conservation of kangaroos important?",
        'answer': "Conservation efforts are important to address threats such as habitat loss and conflicts with human activities, ensuring the continued survival of kangaroo species."
    },
]

# tokenizeng and preparing the dataset
max_length = 512
def prepare_qa_data(data):
    qa_inputs = []

    for example in data:
        inputs = tokenizer(      # Tokenizes the 'question' and 'context' using the BERT tokenizer
            example['question'],
            example['context'],
            truncation='only_second', # truncates only context when above max length
            max_length=max_length,
            padding='max_length', 
            return_tensors='pt' # Returns PyTorch tensors(multi-D array)
        )

        start_positions = tokenizer( # Tokenizes the 'answer' and 'context' using the BERT tokenizer.
            example['answer'],
            example['context'],
            truncation=True,
            max_length=max_length,
            padding='max_length',
            return_tensors='pt'
        )['input_ids'].view(-1).nonzero().squeeze().tolist()[0] # start position of the answer in the tokenized context,  Extracts the index of the first non-padding token as the start position

        end_positions = tokenizer( # Tokenizes the 'context' and 'answer' using the BERT tokenizer.
            example['context'],
            example['answer'],
            truncation=True,
            max_length=max_length,
            padding='max_length',
            return_tensors='pt'
        )['input_ids'].view(-1).nonzero().squeeze().tolist()[-1]  # [input_ids] => This key contains the token IDs of the tokenized input, extracts the index of the last non-padding token, the end position of the answer in the tokenized context.

        qa_inputs.append({
            'input_ids': inputs['input_ids'].squeeze(), # This is a tensor containing the token IDs of the input text.
            'attention_mask': inputs['attention_mask'].squeeze(),  # indicates which tokens are actual input tokens (1) and which are padding tokens (0).
            'start_positions': start_positions, 
            'end_positions': end_positions
        })

    return qa_inputs

qa_inputs = prepare_qa_data(qa_dataset)

# making mini batches for training the model
train_dataloader = DataLoader(qa_inputs, batch_size=8, shuffle=True)

# loss function and optimizer to update the model parameters
optimizer = AdamW(model.parameters(), lr=2e-5) # weight decay, a form of L2 regularization, to prevent overfitting; Weight decay is a regularization technique used during the training of neural networks to prevent overfitting. It is a form of L2 regularization, also known as weight regularization or ridge regularization.
loss_fn = torch.nn.CrossEntropyLoss() #  loss function commonly used in classification tasks. It simplifies the process of computing the cross-entropy loss and is particularly suitable for multi-class classification problems

# fine-tuning the model
num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0

    for batch in tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
        
        inputs = { # extract input
            'input_ids': batch['input_ids'],
            'attention_mask': batch['attention_mask'],
            'start_positions': batch['start_positions'],
            'end_positions': batch['end_positions']
        }

        optimizer.zero_grad() # zero out the gradients of the model parameters of the prev iteration.
        outputs = model(**inputs) # forward pass and generate predictions
        loss = outputs.loss # loss is computed based on the model's output 
        loss.backward() # gradients of the loss with respect to the model parameters during the backward pass
        optimizer.step() # updates the model parameters using the computed gradients

        train_loss += loss.item()

    avg_train_loss = train_loss / len(train_dataloader)
    print(f"Epoch {epoch + 1}/{num_epochs}, Avg Train Loss: {avg_train_loss}")

# Save the fine-tuned model
model.save_pretrained('fine_tuned_model')

while(1):
    question = input("Type your question: ")

    # Tokenizing the input 
    inputs = tokenizer(question, context, return_tensors='pt')

    # Get model outputs
    outputs = model(**inputs) # The **inputs syntax is used to unpack the dictionary inputs and pass its contents as keyword arguments to the model forward method.

    # Get answer start and end indices
    start_index = torch.argmax(outputs.start_logits) # finds the index (position) of the maximum value in the start_logits tensor, start_logits often contains the model's raw predictions or scores for the start position of the answer span, confidence" in the context of start_logits refers to the model's level of certainty or belief in the likelihood that each token in the input sequence corresponds to the starting position of the answer span. This confidence is expressed through the scores assigned to each token in the start_logits tensor.

    end_index = torch.argmax(outputs.end_logits) # In question-answering, the outputs.end_logits tensor typically represents the model's confidence scores for each position in the input sequence as the potential end position of the answer span.Finding the index with the maximum value implies identifying the position where the model predicts the end of the answer span with the highest confidence.

    # Convert indices to tokens and then to string
    answer_tokens = inputs['input_ids'][0][start_index:end_index + 1] # extracting the sequence of tokens from the original input sequence based on the predicted start and end positions, creating a subsequence that the model identifies as the answer span
    answer = tokenizer.decode(answer_tokens) # convert a sequence of token IDs back into a human-readable text string

    print("Answer:", answer)
