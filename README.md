### Create the required environment

`conda env create -f environment.yml`
`conda activate ufo`

### Directory Structure

The directory contains the following folders:

* `data/`: contains some audio samples at `audio/`. The speech chunks get saved at `audio/` and MFCC images get saved at `images/`

* `speech_rec_pytorch/`: contains files to train a digit recognition model from scratch

* `pretrained_model/`: contains pretrained model. Replace with the model you want to use

* `static/` and `templates/`: For the Web App part

* `misc/`: Some utility functions


### Files

**To run the Web App:**
 `export FLASK_APP=application.py` 
 `flask run`. 

Then navigate to `http://127.0.0.1:5000/`.

**To run a simple demo**
`python demo.py -pn <file_name from audio/>`

### Demo

![image.png](contact.png)

