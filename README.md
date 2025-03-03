# Minimal Example for Model Serving through Redis

Before trying this, make sure you have the IP and password of a redis server that can be connected to from both your GPU machine and your local machine. For access to a redis server that works with MPCDF HPC (Raven), you can get access by asking me (Yannik) on slack or through the CHM shared password manager, Bitwarden.

## Example usage

**On GPU machine**
1. Hop on GPU machine (e.g. Raven), clone this and enter the folder.
2. Create a python virtualenv and install the server requirements `pip install -r requirements_server.txt`
3. Grab a GPU. Example: `srun -p gpu --gres=gpu:a100:1 --mem=120GB --time=03:00:00 --pty bash`
4. Set env variables: `export HF_TOKEN=your_hf_access_token;export REDIS_HOST=redis_ip;export REDIS_PASSWORD=replace_with_password`
5. `python serve_model.py "meta-llama/Llama-3.1-8B-Instruct"`. If you get `Cannot access gated repo`, visit huggingface and accept their TOS or swap with a non-gated model (Can also be a path to a local model).

**On local machine**
1. Clone the repo and enter it
2. Create a python virtualenv and install the server requirements `pip install -r requirements_client.txt`
3. Set env variables: `export REDIS_HOST=redis_ip;export REDIS_PASSWORD=replace_with_password`
4. `python model_client.py "Why is the sky blue?" meta-llama/Llama-3.1-8B-Instruct --instruct_format=True`
```
Connecting redis to ■■■.■■■.■■■.■■■
subscribed to 9c15187d-bb47-410b-bba9-17118d86990d
pushing task to Llama-3.1-8B-Instruct
got response from 9c15187d-bb47-410b-bba9-17118d86990d
The sky appears blue because of a phenomenon called scattering, which is the dispersal of light in different directions when it encounters tiny molecules or particles in the atmosphere. This effect is more pronounced in the blue part of the visible spectrum.  When sunlight enters Earth's atmosphere, it encounters tiny molecules of gases such as nitrogen (N2) and oxygen (O2). These molecules are much smaller than the wavelength of light, so they scatter the shorter (blue) wavelengths more than the longer (red) wavelengths.  This scattering effect, known as Rayleigh scattering, is named after the British physicist Lord Rayleigh, who first described it in 187
```