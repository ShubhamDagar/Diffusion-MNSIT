from models import ConditionalDDPM
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import argparse
from utils import seed_everything, compute_fid
from scheduler import NoiseSchedulerDDPM
import os
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import csv

# Add any extra imports you want here

def train(model, train_loader, test_loader, run_name, learning_rate, epochs, batch_size, device, num_steps, masking_schedule="linear"):
    scheduler = NoiseSchedulerDDPM(num_steps, type=masking_schedule, beta_start=0.0001, beta_end=0.02)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    log_file = os.path.join(run_name, "train_log.csv")
    with open(log_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Epoch", "Average Loss"])

    for epoch in range(epochs):
        loop = tqdm(train_loader, leave=True)
        running_loss = 0.0

        for x,y in loop:
            x = x.to(device)
            y = y.to(device)
            t = torch.randint(0, len(scheduler), (x.shape[0], ), device=x.device).long()

            noise = torch.randn_like(x) # generates noise of the same shape as x
            alpha_t = scheduler.alphas.to(device)[t].view(-1, 1, 1, 1)
            x_t = alpha_t.sqrt() * x + (1-alpha_t).sqrt() * noise

            pred_noise = model(x_t, t.long(), y)

            loss = F.mse_loss(pred_noise, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            loop.set_description(f"Epoch [{epoch+1}/{epochs}]")
            loop.set_postfix(loss=loss.item())
        
        avg_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{epochs}] | Avg Loss: {avg_loss:.6f}")

        with open(log_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch + 1, avg_loss])

        if (epoch+1)%5 == 0:
            torch.save(model.state_dict(), f"{run_name}/model_epoch1_{epoch+1}.pth") 
    
    torch.save(model.state_dict(), f"{run_name}/model_final.pth")
    print(f"Training finished!!!")


def sample(model, class_label, device, num_samples=16, num_steps=1000, masking_schedule="linear"):
    '''
    Returns:
        torch.Tensor, shape (num_samples, 1, 28, 28)
    '''
    scheduler = NoiseSchedulerDDPM(num_steps, type=masking_schedule, beta_start=0.0001, beta_end=0.02)
    x_curr = torch.randn(num_samples, 1, 28, 28, device=device)

    with torch.no_grad():
        for t in reversed(range(num_steps)):
            if t > 0:
                z = torch.randn_like(x_curr)
            else:
                z = 0
            alpha_t = scheduler.alphas.to(device)[t].view(-1, 1, 1, 1)
            beta_t = scheduler.betas.to(device)[t].view(-1, 1, 1, 1)
            
            t_batch = torch.full((num_samples,), t, device=device, dtype=torch.long)
            class_label_batch = torch.full((num_samples,), class_label, device=device, dtype=torch.long)
            x_curr = 1/(1-beta_t).sqrt() * (x_curr - (beta_t/(1-alpha_t).sqrt()) * model(x_curr, t_batch, class_label_batch)) + z * beta_t.sqrt()
    
    print("Samples generated!!!")
    return x_curr


def parse_args():
    parser = argparse.ArgumentParser(description="DDPM Conditional Model Template")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--num_steps", type=int, default=1000, help="Number of diffusion steps")
    parser.add_argument("--num_samples", type=int, default=16, help="Number of samples to generate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "sample"], help="Mode: train or sample")
    parser.add_argument("--masking_schedule", type=str, default="linear", choices=["linear", "cosine"], help="Masking Schedule: linear or cosine")
    # Add any other arguments you want here
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    if device.type == "cuda":
        print("GPU Name:", torch.cuda.get_device_name(0))

    ### Data Preprocessing Start ### (Do not edit this)
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    ### Data Preprocessing End ### (Do not edit this)

    sample_img, _ = train_dataset[0]
    in_channel = sample_img.shape[0]
    model = ConditionalDDPM(in_channel=in_channel, out_channel=in_channel, num_classes=10)
    model.to(device)

    run_name = f"exps_ddpm/final/conditional_ddpm_{args.epochs}ep_{args.batch_size}bs_{args.learning_rate}lr_{args.num_steps}num_steps_{args.masking_schedule}_masking_schedule" # Change run name based on your experiments
    os.makedirs(run_name, exist_ok=True)

    if args.mode == "train":
        model.train()
        train(model, train_loader, test_loader, run_name, args.learning_rate, args.epochs, args.batch_size, device, args.num_steps, args.masking_schedule)
    elif args.mode == "sample":
        model.load_state_dict(torch.load(f"{run_name}/model_final.pth"))
        model.eval()
        for class_num in range(10):
            samples = sample(model, class_num, device, args.num_samples, args.num_steps, args.masking_schedule)
            torch.save(samples, f"{run_name}/{class_num}class_{args.num_samples}samples_{args.num_steps}steps.pt")
