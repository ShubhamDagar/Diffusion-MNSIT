Command used for training

CUDA_VISIBLE_DEVICES=0 python3 ddpm.py --epochs 25 --batch_size 512 --num_steps 1000 --masking_schedule linear

Command To Run the generator.py

`CUDA_VISIBLE_DEVICES=0 python3 generator.py --diffusion conditional_ddpm --masking_schedule cosine --generate_images 1 --num_steps 2500`

Here,
--diffusion -> means type of diffusion and has entries: ["ddpm", "conditional_ddpm"]
--masking_schedule -> linear or cosine
--generate_images -> it is as flag in case you want to see the generated samples
--num_steps -> number of time steps

To run this command for a specific model, add the hyperparamters used for the training of that specific model as arguments.
The corresponding FID scores would be saved in a .csv file, named as fid_results.csv

Also, the model folder structure should be like this:

generator.py
exps_ddpm
        -final
            -model weight folder