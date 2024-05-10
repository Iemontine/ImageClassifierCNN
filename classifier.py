import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
from PIL import Image
import os
import shutil

# Normalize the data such that pixel values are floats in [0, 1]
transform = transforms.Compose([ transforms.ToTensor(), transforms.Normalize((0.0,), (1.0,)) ])

# Load training and test datasets
train_dataset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

# Split training data into training and validation sets
validation_size = 12000
train_size = len(train_dataset) - validation_size
train_data, val_data = random_split(train_dataset, [train_size, validation_size])

# Dataloaders for training, validation, and test sets, 32 observations per batch
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader = DataLoader(val_data, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Define the convolutional neural network model
class CNN(nn.Module):
	def __init__(self):
		super(CNN, self).__init__()
		# 2D convolutional layers
		self.layer1 = nn.Sequential(	
			nn.Conv2d(1, 28, 3),		# 1 input channel, 28 filters, 3x3 window size
			nn.ReLU(),					# ReLU activation
			nn.MaxPool2d(2, 2) 			# 2x2 max pooling
		)
		self.layer2 = nn.Sequential(
			nn.Conv2d(28, 56, 3),		# 28 input channels, 56 filters, 3x3 window size
			nn.ReLU(),					# ReLU activation
		)
		# Fully connected layers
		self.fc1 = nn.Linear(56 * 11 * 11, 56)		# 56x1x11 input nodes, 56 output nodes
		self.relu = nn.ReLU() 						# ReLU activation
		self.fc2 = nn.Linear(56, 10)				# 56 input nodes, 10 output nodes
		self.softmax = nn.Softmax(dim=1)			# Softmax activation

	# Forward pass
	def forward(self, x):
		x = self.layer1(x) 				# Pass through first convolutional layer
		x = self.layer2(x)				# Pass through second convolutional layer
		x = x.view(-1, 56 * 11 * 11)	# Flatten the output from the convolutional layers
		x = self.relu(self.fc1(x))		# Pass through first fully connected layer with ReLU activation
		x = self.softmax(self.fc2(x))	# Pass through second fully connected layer with softmax activation
		return x

# Initialize the model, sparse categorical cross-entropy loss function, and Adam optimizer
model = CNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())
# Used to track training and validation accuracies across epochs
train_accuracies = []
val_accuracies = []
num_epochs = 10

if os.path.exists('model.pth'):
	model.load_state_dict(torch.load('model.pth'))
	print("Model loaded, skipping training.")
else:
	for epoch in range(num_epochs): # Train the model for 10 epochs
		# Training Phase
		model.train()
		running_loss = 0.0
		correct_train = 0
		total_train = 0
		for inputs, labels in train_loader:	
			optimizer.zero_grad()				# Zero the parameter gradients
			outputs = model(inputs)				# Forward pass
			loss = criterion(outputs, labels)	# Compute loss
			loss.backward()						# Backward pass
			optimizer.step()					# Optimize the model

			_, predicted = torch.max(outputs.data, 1)			# Get predicted labels
			total_train += labels.size(0)						# Track total observations
			correct_train += (predicted == labels).sum().item() # Track correct predictions
			running_loss += loss.item()							# Track running loss
		train_accuracy = 100 * correct_train / total_train		# Training accuracy
		
		# Validation Phase
		model.eval()
		correct_val = 0
		total_val = 0
		for inputs, labels in val_loader:
			outputs = model(inputs)								# Forward pass
			_, predicted = torch.max(outputs.data, 1)			# Get predicted labels
			total_val += labels.size(0)							# Track total observations
			correct_val += (predicted == labels).sum().item()	# Track correct predictions
		val_accuracy = 100 * correct_val / total_val			# Validation accuracy

		train_accuracies.append(train_accuracy)
		val_accuracies.append(val_accuracy)
		print(f"Epoch {epoch+1}: Loss = {running_loss:.3f}, Train Accuracy = {train_accuracy:.2f}%, Val Accuracy = {val_accuracy:.2f}%")
	torch.save(model.state_dict(), 'model.pth')

# Evaluate on test set
correct_test = 0
total_test = 0
test_examples = {}
model.eval()
with torch.no_grad():										# Disable gradient tracking for evaluation
	for inputs, labels in test_loader:						# Iterate over test set
		outputs = model(inputs)								# Forward pass
		_, predicted = torch.max(outputs.data, 1)			# Get predicted labels
		total_test += labels.size(0)						# Track total observations
		correct_test += (predicted == labels).sum().item()	# Track correct predictions
		for i, pred in enumerate(predicted):				# Track misclassified examples
			if labels[i] not in test_examples and pred != labels[i]:
				test_examples[labels[i].item()] = (inputs[i], pred, labels[i])
test_accuracy = 100 * correct_test / total_test				# Test accuracy

# Save misclassified examples
shutil.rmtree("misclassified")
os.makedirs("misclassified")
for label, (img, pred, true_label) in test_examples.items():
	if pred != true_label:
		img_path = f"misclassified/True{true_label}_Pred{pred}_{label}.png"
		Image.fromarray(img.squeeze().numpy() * 255).convert('L').save(img_path)

# Model performance evaluation
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("============Model Performance===========")
print(f"Number of Trainable Parameters: {trainable_params}")
print(f"Test Accuracy: {test_accuracy:.2f}%")
print(f"Number of Misclassified Examples: {len(test_examples)}")

# Evaluate training and validation accuracy at the end of each epoch, and plot them as line plots
plt.figure(figsize=(num_epochs, 5))
plt.plot(range(num_epochs), train_accuracies, label='Training Accuracy', color='red')
plt.plot(range(num_epochs), val_accuracies, label='Validation Accuracy', color='blue')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy per Epoch')
plt.legend()
plt.text(num_epochs-1, train_accuracy, f"Train Acc: {train_accuracy:.2f}%", ha='right', va='bottom', color='red')
plt.text(num_epochs-1, val_accuracy, f"Val Acc: {val_accuracy:.2f}%", ha='right', va='bottom', color='blue')
plt.show()