LCM-LoRA makes it possible for you to use diffusion models in significantly less steps. 

This repo was built and tested on an Apple M1 with Python 3.9  

### Setup:
* Optionally setup a virtual environment: `python -m venv env && source ./env/bin/activate`
* Install the deps: `pip install -r requirements.txt`
* Run the main script to bring up the UI: `python main.py`
* Nvidia users: ```pip install torch --extra-index-url https://download.pytorch.org/whl/cu121```

The UI should stand up on `http://localhost:7860/`


![lcm-lora-demo.gif](img%2Flcm-lora-demo.gif)

[We took inspiration from this repo](https://github.com/flowtyone/flowty-realtime-lcm-canvas) and [Hugging Face Docs](https://huggingface.co/docs/diffusers/main/en/using-diffusers/inference_with_lcm_lora)
