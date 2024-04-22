# Website target action predictor bot
This repository contains the project, a Telegram bot that predicts session outcomes based on input data.

### Deployment
The project can be deployed through telegram bot.

### Configuration
To run the project, you'll need to replace the Telegram token in modules/config.py with your own. The bot expects input in the format of a string, similar to the ones found in dataset ga_sessions.csv. For example:

9056903658751762760.1624612805.1624612805,2108724708.1624613192,2021-06-25,12:00:00,1,MvfHsxITijuriZxsqZqt,cpv,FTjNLDyTrXaWYgZymFkV,PkybGvWbaqORmxjNunqZ,,mobile,,Samsung,,412x846,Samsung Internet,Russia,Moscow

The bot will respond with a prediction in the format session_id: prediction (0 or 1).

### Running the Project
To run the project without retraining the model (the trained model is located in data/models), simply execute the modules/bot.py script. However, you'll need to create a virtual environment in the root directory and install all dependencies with <b>pip install -r requirements.txt</b> in your terminal.

If you need to test the pipeline for training the model, you should place the ga_hits.csv and ga_sessions.csv files in the data/train directory and run the modules/pipeline.py script. Dataset is not included in this repo.

### Research and Data Analysis
A Jupyter notebook containing data analysis and processing is located in the research/ directory.

Note: The ga_hits.csv and ga_sessions.csv files are not included in this repository due to their large size and possible NDA.

### Extra
Project is made due to educational purposes only. The main idea is to study how to build a pipeline of data processing, transformation, modelling, prediction and wrap it all into telegram bot.
Original data from: Google Analytics (last-click attribution model)
