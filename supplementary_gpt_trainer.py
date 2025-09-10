"""

Supplementary code file to train GPT2 
on Witten's papers. 

This file has two important functions. 

(1) It has a function to preprocess
the Witten's dataset. 

(2) It has a code to implement 
a custom Trainer object that we use to 
run the training loop. 

"""

from transformers import get_scheduler
import torch
from torch.optim import AdamW

def get_tokenized_data(tokenizer, data_file, batch_size, context_length, shuffle = False):

  '''
    Returns: Tokenized dataset as a Torch Tensor of
             shape (N, B, C) where
             B = batch_size
             C = context_length
             N = number of batches (determined by the size of data_file).
  '''

  # Initializing the dataset
  tokenized_data = tokenizer(data_file)['input_ids']

  # Removing the last few lines from the data
  # to ensure we have integer multiple of
  # context length and batch_size worth of data
  no_batches = len(tokenized_data) // (context_length * batch_size)
  data_size = context_length * batch_size * no_batches
  tokenized_data = tokenized_data[:data_size]

  # Converting into Torch tensor
  tokenized_data = torch.tensor(tokenized_data)

  # Reshaping the tensor
  tokenized_data = tokenized_data.view(-1,context_length)

  # Shuffle
  if shuffle:
    perm_indices = torch.randperm(tokenized_data.size(0))
    tokenized_data = tokenized_data[perm_indices]

  # Reshaping again
  tokenized_data = tokenized_data.view((-1,batch_size,context_length))

  return tokenized_data

class gpt_trainer():

  def __init__(self, model, train_data, val_data, device, optimizer_config):

    '''
      Args:
          model - pretrained GPT2 model to be finetrained
          train_data - Torch tensor of the shape (num_batches, batch_size, context_length)
          val_data - Torch tensor of the shape (num_batches, batch_size, context_length)
          device - cuda or CPU
          optimizer_config: dict containing epochs_list,
                                            optimizer,
                                            learning_rate,
                                            and final_learning_rate.
    '''

    # Initializing the model
    self.model = model
    self.device = device
    self.model.to(device)

    # Initializing the data
    self.train_data = train_data
    self.val_data = val_data

    # Initializing the optimizer and training settings
    init_lr = optimizer_config['learning_rate']
    self.optimizer = optimizer_config['optimizer'](self.model.parameters(),
                                                   lr = init_lr,
                                                   )
    self.epochs = optimizer_config['epochs_list']
    self.num_batches = train_data.size(0)
    self.final_lr = optimizer_config['final_lr']
    
    # Initializing the lr scheduler (if needed)
    if self.final_lr == None:

        self.lr_scheduler = False
    
    else:
        
        self.lr_scheduler = True

        # get_scheduler is supposed to take the
        # number of training steps and it reduces
        # the lr to zero by that number of steps.
        # However, we want the final lr to be 
        # non-zero, so we need to scale the number 
        # of steps accordingly.
        num_steps = len(self.epochs)*self.num_batches
        scaled_num_steps = num_steps * init_lr / (init_lr - self.final_lr)
        
        self.scheduler = get_scheduler(
            "linear",
            self.optimizer,
            num_warmup_steps=0,
            num_training_steps=scaled_num_steps,
        )

    # Initializing the history of losses during training
    self.train_loss_history = []
    self.val_loss_history = []


  def run(self):

    self.model.train()

    for epoch in self.epochs:

      for batch in range(self.num_batches):

        batch_data = self.train_data[batch]   # Tensor shape (batch_size, context_length)
        output = self.model.forward(batch_data.to(self.device), labels=batch_data.to(self.device))
        loss = output.loss
        loss.backward()

        # Gradient step
        self.optimizer.step()
        if self.lr_scheduler:
          self.scheduler.step()
        self.optimizer.zero_grad()

        if (batch % 50 == 0) or (batch == (self.num_batches-1)):
          train_loss , val_loss = self.estimate_losses()
          print(f"Epoch {epoch}, step {batch}: training loss is {train_loss} and validation loss is {val_loss}")
          self.train_loss_history.append(train_loss)
          self.val_loss_history.append(val_loss)

      file_path = f"trained_model_weights_after_epoch_{epoch}.pth"
      self.save_model(file_path)

    self.model.eval()

  def estimate_losses(self):

    '''
      Return : estimate of train loss and val loss
    '''

    self.model.eval()
      
    train_loss = 0.0
    val_loss = 0.0

    # Since val_data is smaller than than the train_data,
    # we will use the val_data size subset of the train_data
    # to estimate the train loss. We will take samples of
    # train data at equal intervals.
    ratio = int(self.train_data.size(0) / self.val_data.size(0))
    
    num_val_batches = self.val_data.size(0)
    for batch in range(num_val_batches):

      # Train loss
      train_batch = self.train_data[ratio*batch]
      output = self.model.forward(train_batch.to(self.device), labels=train_batch.to(self.device))
      train_loss += output.loss.item()

      # Val loss
      val_batch = self.val_data[batch]
      output = self.model.forward(val_batch.to(self.device), labels=val_batch.to(self.device))
      val_loss += output.loss.item()

    train_loss = train_loss / num_val_batches
    val_loss = val_loss / num_val_batches

    self.model.train()
      
    return train_loss, val_loss

  def save_model(self, file_path):

    # saving only the weights of the models
    torch.save(self.model.state_dict(), file_path)
