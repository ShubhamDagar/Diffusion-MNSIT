import os
import shutil
import torch
import matplotlib.pyplot as plt
from ddpm import sample as sample_ddpm
from ddpm_cond import sample as sample_ddpm_cond
from utils import compute_fid
import csv
import argparse
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from models import DDPM
from models import ConditionalDDPM
import random

def generate_images_from_samples(output_dir, samples):
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    for i in range(samples.shape[0]):
        plt.imshow(samples[i, 0].detach().cpu().numpy(), cmap='gray')
        plt.axis('off')
        plt.savefig(os.path.join(output_dir, f"sample_{i}.png"))
        plt.close()

def get_real_images_with_given_class_label(test_dataset, class_num, batch_size=64):
    indices = [i for i, (_, label) in enumerate(test_dataset) if label == class_num]
    selected_indices = torch.randperm(len(indices))[:batch_size]
    return torch.stack([test_dataset[indices[i]][0] for i in selected_indices])

def get_two_real_batches(test_loader, device):
    idx1, idx2 = random.sample(range(len(test_loader)), 2)
    batch1, _ = list(test_loader)[idx1]
    batch2, _ = list(test_loader)[idx2]
    return batch1.to(device), batch2.to(device)

def calculate_fid_for_samples(test_dataset, test_loader, model, device, num_samples, num_steps, masking_schedule, conditional=False, class_num=None, iterations=3, ideal=False):
    temp = 0
    if conditional:
        iterations=1
        
    for i in range(iterations):
        if conditional:
            if ideal:
                real_samples_1 = get_real_images_with_given_class_label(test_dataset, class_num)
                real_samples_2 = get_real_images_with_given_class_label(test_dataset, class_num)
                temp += compute_fid(real_samples_1, real_samples_2)
            else:
                generated_samples = sample_ddpm_cond(model, class_num, device, num_samples, num_steps, masking_schedule)
                generated_samples = (generated_samples + 1) / 2
                generated_samples = torch.clamp(generated_samples, 0, 1)
                real_samples = get_real_images_with_given_class_label(test_dataset, class_num)
                temp += compute_fid(real_samples, generated_samples)
        else:
            if ideal:
                real_samples_1, real_samples_2 = get_two_real_batches(test_loader, device)
                temp += compute_fid(real_samples_1, real_samples_2)
            else:
                generated_samples = sample_ddpm(model, device, num_samples, num_steps, masking_schedule)
                generated_samples = (generated_samples + 1) / 2
                generated_samples = torch.clamp(generated_samples, 0, 1)
                real_samples,_ = next(iter(test_loader))
                temp += compute_fid(real_samples, generated_samples)
    temp /= iterations
    return temp

def parse_args():
    parser = argparse.ArgumentParser(description="Samples Generator")
    parser.add_argument("--diffusion", type=str, default="ddpm", choices=["ddpm", "conditional_ddpm"], help="Type of diffusion")
    parser.add_argument("--epochs", type=int, default=25, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--num_steps", type=int, default=1000, help="Number of diffusion steps")
    parser.add_argument("--num_samples", type=int, default=64, help="Number of samples to generate")
    parser.add_argument("--masking_schedule", type=str, default="linear", choices=["linear", "cosine"], help="Masking Schedule: linear or cosine")
    parser.add_argument("--generate_images", type=int, default=0, choices=[0, 1], help="See the generated samples")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    if device.type == "cuda":
        print("GPU Name:", torch.cuda.get_device_name(0))

    run_name = f"exps_ddpm/final/{args.diffusion}_{args.epochs}ep_{args.batch_size}bs_{args.learning_rate}lr_{args.num_steps}num_steps_{args.masking_schedule}_masking_schedule" # Change run name based on your experiments
    
    fid_csv_file = "fid_results.csv"
    columns = ["diffusion_type", "batch_size", "epochs", "num_steps", "learning_rate", "masking_schedule", "class_num", "fid_score", "ideal_value"]
    if not os.path.exists(fid_csv_file):
        with open(fid_csv_file, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(columns)

    # dataset loading
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    ####

    sample_img, _ = train_dataset[0]
    in_channel = sample_img.shape[0]
    
    if args.diffusion == "ddpm":
        model = DDPM(in_channel=in_channel, out_channel=in_channel)
    else:
        model = ConditionalDDPM(in_channel=in_channel, out_channel=in_channel, num_classes=10)

    model.to(device)
    model.load_state_dict(torch.load(f"{run_name}/model_final.pth"))
    model.eval()

    if args.diffusion == "ddpm":
        samples = sample_ddpm(model, device, args.num_samples, args.num_steps, args.masking_schedule)
        torch.save(samples, f"{run_name}/samples_ddpm.pt")

        if args.generate_images:
            output_dir = f"sample_images/ddpm/{run_name}"
            generate_images_from_samples(output_dir, samples)
        
        fid_score = calculate_fid_for_samples(train_dataset, train_loader, model, device, args.num_samples, args.num_steps, args.masking_schedule, ideal=True)
        with open(fid_csv_file, mode="a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([args.diffusion, args.batch_size, args.epochs, args.num_steps, args.learning_rate, args.masking_schedule, "None", fid_score, "yes"])
    
        fid_score = calculate_fid_for_samples(test_dataset, test_loader, model, device, args.num_samples, args.num_steps, args.masking_schedule)
        with open(fid_csv_file, mode="a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([args.diffusion, args.batch_size, args.epochs, args.num_steps, args.learning_rate, args.masking_schedule, "None", fid_score, "no"])

    else:
        for class_num in range(10):
            samples = sample_ddpm_cond(model, class_num, device, args.num_samples, args.num_steps, args.masking_schedule)
            torch.save(samples, f"{run_name}/samples_ddpm_cond_{class_num}.pt")

            if args.generate_images:
                output_dir = f"sample_images/ddpm/{run_name}/{class_num}"
                generate_images_from_samples(output_dir, samples)

            fid_score = calculate_fid_for_samples(train_dataset, train_loader, model, device, args.num_samples, args.num_steps, args.masking_schedule, conditional=True, class_num=class_num, ideal=True)
            with open(fid_csv_file, mode="a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([args.diffusion, args.batch_size, args.epochs, args.num_steps, args.learning_rate, args.masking_schedule, class_num, fid_score, "yes"])
        
            fid_score = calculate_fid_for_samples(test_dataset, test_loader, model, device, args.num_samples, args.num_steps, args.masking_schedule, conditional=True, class_num=class_num)
            with open(fid_csv_file, mode="a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([args.diffusion, args.batch_size, args.epochs, args.num_steps, args.learning_rate, args.masking_schedule, class_num, fid_score, "no"])
    
    print("fid calculation is done!!!")