# nu
import json
with open("../preprocess/arg_split.json") as f:
    arg_data = json.load(f)

arg_tokenized_text = tokenize_all_text(embed_lookup, arg_data['train'])
arg_test_tokenized_text = tokenize_all_text(embed_lookup, arg_data['test'])

# Test your implementation!

seq_length = 15

arg_train_features = pad_features(arg_tokenized_text, seq_length=seq_length)

## test statements - do not change - ##
assert len(arg_train_features)==len(arg_tokenized_text), "Features should have as many rows as reviews."
assert len(arg_train_features[0])==seq_length, "Each feature row should contain seq_length values."

# print first 8 values of the first 20 batches 
print(arg_train_features[:20,:8])

# Test your implementation!

arg_test_features = pad_features(arg_test_tokenized_text, seq_length=seq_length)

## test statements - do not change - ##
assert len(arg_test_features)==len(arg_test_tokenized_text), "Features should have as many rows as reviews."
assert len(arg_test_features[0])==seq_length, "Each feature row should contain seq_length values."

# print first 8 values of the first 20 batches 
print(arg_test_features[:20,:8])


arg_train_labels = np.array([item['label'] for item in arg_data['train'].values()])
arg_test_labels = np.array([item['label'] for item in arg_data['test'].values()])

print(arg_test_labels[:20])

split_frac = 0.8

## split data into training, validation, and test data (features and labels, x and y)

split_idx = int(len(arg_train_features)*split_frac)
arg_train_x, arg_valid_x = arg_train_features[:split_idx], arg_train_features[split_idx:]
arg_train_y, arg_valid_y = arg_train_labels[:split_idx], arg_train_labels[split_idx:]

## print out the shapes of your resultant feature data
print("\t\t\tFeature Shapes:")
print("Train set: \t\t{}".format(arg_train_x.shape), 
      "\nValidation set: \t{}".format(arg_valid_x.shape),
      "\nTest set: \t\t{}".format(arg_test_features.shape))

from torch.utils.data import TensorDataset, DataLoader

# create Tensor datasets
arg_train_data = TensorDataset(torch.from_numpy(arg_train_x), torch.from_numpy(arg_train_y))
arg_valid_data = TensorDataset(torch.from_numpy(arg_valid_x), torch.from_numpy(arg_valid_y))
arg_test_data = TensorDataset(torch.from_numpy(arg_test_features), torch.from_numpy(arg_test_labels))

# dataloaders
batch_size = 4

# shuffling and batching data
arg_train_loader = DataLoader(arg_train_data, shuffle=True, batch_size=batch_size)
arg_valid_loader = DataLoader(arg_valid_data, shuffle=True, batch_size=batch_size)
arg_test_loader = DataLoader(arg_test_data, shuffle=True, batch_size=batch_size)

# nu lstm

vocab_size = len(pretrained_words)
output_size = 3 # binary class (1 or 0)
embedding_dim = len(embed_lookup[pretrained_words[0]]) # 300-dim vectors

hidden_dim = 128

batch_size = 4
seq_length = 15

num_layers =2

arg_lstm = BiLSTMSentiment(embed_lookup, vocab_size, output_size, embedding_dim,
                   hidden_dim, batch_size, seq_length, num_layers)

print(arg_lstm)

# loss and optimization functions
lr=0.001

criterion = nn.CrossEntropyLoss()
arg_optimizer = torch.optim.Adam(arg_lstm.parameters(), lr=lr, weight_decay = 0.01)



test_losses = [] # track loss
num_correct = 0
pred_tensor = None
label_tensor = None


arg_lstm.eval()
# iterate over test data
for inputs, labels in arg_test_loader:

    
    # get predicted outputs
    output = arg_lstm(inputs)
    
    # calculate loss
    test_loss = criterion(output, labels)
    test_losses.append(test_loss.item())
    
    # convert output probabilities to predicted class (0 or 1)
    pred = torch.argmax(output, dim=1)  # argmax

    # compare predictions to true label
    correct_tensor = pred.eq(labels.view_as(pred))
    correct = np.squeeze(correct_tensor.numpy())
    num_correct += np.sum(correct)

    if pred_tensor == None:
        pred_tensor = pred
    else:
        pred_tensor = torch.cat((pred_tensor, pred), dim=-1)

    if label_tensor == None:
        label_tensor = labels
    else:
        label_tensor = torch.cat((label_tensor, labels), dim=-1)

from torcheval.metrics.functional import multiclass_f1_score
from torcheval.metrics.functional import multiclass_precision
from torcheval.metrics.functional import multiclass_recall

print("micro F1: {:.3f}".format(multiclass_f1_score(pred_tensor, label_tensor, num_classes=3).item()))
print("macro F1: {:.3f}".format(multiclass_f1_score(pred_tensor, label_tensor, num_classes=3, average='macro').item()))
print("weighted F1: {:.3f}\n".format(multiclass_f1_score(pred_tensor, label_tensor, num_classes=3, average='weighted').item()))

print("micro precision: {:.3f}".format(multiclass_precision(pred_tensor, label_tensor, num_classes=3).item()))
print("macro precision: {:.3f}".format(multiclass_precision(pred_tensor, label_tensor, num_classes=3, average='macro').item()))
print("weighted precision: {:.3f}\n".format(multiclass_precision(pred_tensor, label_tensor, num_classes=3, average='weighted').item()))

print("micro recall: {:.3f}".format(multiclass_recall(pred_tensor, label_tensor, num_classes=3).item()))
print("macro recall: {:.3f}".format(multiclass_recall(pred_tensor, label_tensor, num_classes=3, average='macro').item()))
print("weighted recall: {:.3f}\n".format(multiclass_recall(pred_tensor, label_tensor, num_classes=3, average='weighted').item()))
