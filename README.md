# Sudoku Solver 
Keras neural network trained to solve sudoku puzzles.

## Usage 
Set `BASE_PATH` to your machine's proper location of the repo at the top of [`model.py`](/model.py).

## Dataset
The dataset used to train the model is a csv file containing 1 million puzzles with one column named __quizzes__ and another column named __solutions__. The download link and description to this dataset can be found [here](https://www.kaggle.com/bryanpark/sudoku/version/3). The data preparation process I followed can also be found in `data_prep.py` under the [`get_data()`](/data_prep.py) function. 

### TODO
- Get custom loss function working
