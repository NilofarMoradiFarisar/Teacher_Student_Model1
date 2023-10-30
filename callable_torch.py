import torch.nn as nn
import TeacherModel, student_model, model from student_teacher
# Define the loss function
def my_loss_fn(output, target):
    loss = nn.MSELoss()(output, target)
    return loss
num_epochs = 10
optimizer = optim.SGD(student_model.parameters(), lr=0.1)

# Use the loss function in the training loop
for i in range(num_epochs):
    optimizer.zero_grad()
    output = model(input)
    loss = my_loss_fn(output, target)
    loss.backward()
    optimizer.step()