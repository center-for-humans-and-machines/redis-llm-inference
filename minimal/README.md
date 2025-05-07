# Minimal example of how to use the redis relay
## A priori requirements
* You have the redis login data (REDIS_HOST, REDIS_PASSWORD)
* You have the redis-relay api key (RELAY_API_KEY)
* The redis-relay deployment location (REDIS_RELAY=https://redis-relay.chm.mpib-berlin.mpg.de)
* If you plan to use a huggingface model, create a huggingface user access token (HF_TOKEN)
## The deployment side of things (Raven)
1. Log in to raven
2. Clone `https://github.com/center-for-humans-and-machines/redis-llm-inference` using git. Enter the `minimal` folder.
3. Grab a GPU shell `srun -p gpu --gres=gpu:a100:1 --mem=120GB --time=05:30:00 --pty bash`. Usually takes less than 30s. Could take longer if raven is overloaded.
    * Verify that you are on a gpu shell using `nvidia-smi`
4. Select a suitable apptainer image to start the deployment script. You may use `/u/yjiang/projects/coopbot/llm-strategic-tuning/images/strategic_v11.sif`
5. `apptainer exec --nv --contain --cleanenv --pwd /root/minimal --bind .:/root/minimal --bind ~/.cache/huggingface:/root/.cache/huggingface --env HF_TOKEN=[Your HF user token] --env HF_HOME="/root/.cache/huggingface" --env REDIS_HOST=[the redis hostname] --env REDIS_PASSWORD=[the redis password] /u/yjiang/projects/coopbot/llm-strategic-tuning/images/strategic_v11.sif python minimal_model_deploy.py [Your Model Name]`
    * Make sure you replace all `[]` text appropiately
    * Your model name could be something like `meta-llama/Meta-Llama-3-8B-Instruct`
6. The model may take some time to load (up to a few minutes). If you see `Successfully connected to Redis and now listening for tasks on queue: [Last part of your model name]`, you are all set.
    * The last part of the model name will be the queue that your model will process.
## Sending requests to the deployed model
1. On your local machine, clone `https://github.com/center-for-humans-and-machines/redis-llm-inference` using git. Enter the `minimal` folder.
2. Install requests `pip install requests` (you may use a virtualenv)
3. Set your env variables `export REDIS_RELAY=https://redis-relay.chm.mpib-berlin.mpg.de;export RELAY_API_KEY=[The relay api key]`
4. `python minimal_submit_to_redis_relay.py --prompt "Why is the sky blue" --model_name Meta-Llama-3-8B-Instruct`
    * The model name is the name of the queue