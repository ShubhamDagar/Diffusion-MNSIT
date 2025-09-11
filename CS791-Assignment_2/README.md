# Environment and installations

To install Miniconda, follow the steps [here](https://www.anaconda.com/docs/getting-started/miniconda/install).

To setup the environment, follow these steps:

```
conda create --name cs791env python=3.8 -y
conda activate cs791env
```

Install the dependencies (if any):
```
pip install -r requirements.txt
```

To install torch, you can follow the steps [here](https://pytorch.org/get-started/locally/). You'll need to know the cuda version on the server. Use `nvitop` command to know the version first. If you have cuda version 12.4, you can just do:
```
pip install torch
```

To check if GPU is connected, run the command.
```
print("CUDA available:", torch.cuda.is_available())
```

In case multiple GPUs are present in the system, we recommend using the environment variable `CUDA_VISIBLE_DEVICES` when running your scripts. For example, below command ensures that your script runs on 7th GPU. 
```
CUDA_VISIBLE_DEVICES=7 python d3pm_template.py --mode train
```

CUDA error messages can often be cryptic and difficult to debug. In such cases, the following command can be quite useful:
```
CUDA_VISIBLE_DEVICE=-1 python d3pm_template.py --mode train
```
This forces the script to run exclusively on the CPU.



# Server Connection

1. In VS Code, install Remote – SSH extension. Make sure OpenSSH client is installed on your system.

2. Open Remote Explorer in VS Code.

Click "+" and select Add New SSH Host.
Enter server details in the format:
username@hostname
Example:
alice@gpu1.cse.iitb.ac.in

OR,

edit ~/.ssh/config directly:
Host gpu1.cse.iitb.ac.in
    HostName gpu1.cse.iitb.ac.in
    User alice

3. In Remote Explorer, click your saved server (e.g. server1). VS Code will prompt for the password. Once authenticated, a new VS Code window opens connected to the server.

For more details, visit, https://code.visualstudio.com/docs/remote/ssh.

To forcefully shutdown kernel, use these commands:
```
import os
os._exit(0)
```


