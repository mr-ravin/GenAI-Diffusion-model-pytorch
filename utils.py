import torch
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd 

def save_plot(train_loss_list=[], filter_bucket=1):
        plt.figure()
        red_patch = mpatches.Patch(color='red', label='Train Loss')
        x_axis_data = list(range(1,len(train_loss_list)+1))
        x_axis_data = [x * filter_bucket for x in x_axis_data]
        sns.lineplot(x=x_axis_data, y=train_loss_list, color='red', alpha=0.75)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Denoising Diffusion Model Training Analysis")
        plt.legend(handles=[red_patch], loc='upper right')
        plt.savefig('./result/training_analysis.png')
        print("Saved ./result/training_analysis.png")


def linear_beta_schedule(timesteps=300, start=0.0001, end=0.02):
    return torch.linspace(start, end, timesteps)

def get_value_at_time_t(all_values, t, x_shape):
    values_at_t = all_values.gather(-1, t.cpu())
    batch_size = t.shape[0] 
    return values_at_t.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)
