import matplotlib.pyplot as plt
import pandas as pd

def loss_averaging(data):
    # Convert relevant columns to numeric for aggregation (if needed)
    data[['G_GAN', 'G_L1', 'D_real', 'D_fake']] = data[['G_GAN', 'G_L1', 'D_real', 'D_fake']].apply(pd.to_numeric, errors='coerce')

    # Group by 'Epoch' and calculate the mean for the specified columns
    average_data_per_epoch = data.groupby('Epoch')[['G_GAN', 'G_L1', 'D_real', 'D_fake']].mean().reset_index()

    # Sort the DataFrame by 'Epoch'
    average_data_per_epoch.sort_values('Epoch', inplace=True)

    # Save the result to a new CSV file
    average_data_per_epoch.to_csv('average_data_per_epoch.csv', index=False)

    return average_data_per_epoch

def plot(avg_data):
    plt.figure(figsize=(10, 5))

    # Subplot 1: G_GAN loss
    plt.subplot(2, 2, 1)
    plt.plot(avg_data['G_GAN'])
    plt.title('Generator Loss GAN')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    # Subplot 2: D_GAN loss
    plt.subplot(2, 2, 2)
    plt.plot(avg_data['G_L1'])
    plt.title('Generator Loss L1')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    # Subplot 3: D_Loss
    plt.subplot(2, 2, 3)
    plt.plot(avg_data['D_real'])
    plt.title('Discriminator Loss Real')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    # Subplot 4: G_Loss
    plt.subplot(2, 2, 4)
    plt.plot(avg_data['D_fake'])
    plt.title('Discriminator Loss Fake')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.tight_layout()
    plt.savefig('graph.png')
    plt.show()

if __name__ == '__main__':
    # Load the CSV file with the correct delimiter (semicolon)
    data = pd.read_csv('loss.csv', delimiter=';')

    # check if the data has the correct columns: 'G_GAN', 'G_L1', 'D_real', 'D_fake'
    if 'G_GAN' in data.columns and 'G_L1' in data.columns and 'D_real' in data.columns and 'D_fake' in data.columns:
        average_data_per_epoch = loss_averaging(data)
        plot(average_data_per_epoch)
    else:
        print('The data does not contain the required columns: G_GAN, G_L1, D_real, D_fake')