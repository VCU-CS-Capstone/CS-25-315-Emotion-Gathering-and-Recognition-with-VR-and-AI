
To run code you need to to 
create a virtual environment
pip install numpy
pip install opencv
pip install mediapipe
pip install scikit-learn

To run program, run test_model.py

If you want to run different data create a "data" folder inside same directory with data inside designate folder (example data/happy/... or data/sad/...)
When you add data you have to run prepare_data.py and it will create a new data.txt file.  Becareful to not use too much data on your personal computer.
After it is prepared.  Run train_model which will create a model file and give the accuracy and accuracy matrix of the model.
Now you can run test_model.py
