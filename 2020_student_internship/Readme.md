# Visualisation for neural networks
Tries to approximate the function `w[0] + w[1]x + w[2]x² + w[3]x³ + ...`
## Setup
```
pip install numpy
pip install matplotlib
```
## Usage
Change the values for `Ws`, `len_w`, `batches`, `batch_size`, `lr`, `momentum` and `points`, to get the results you want
### Ws
The w's for the target-function (see above). Set to `[random.random()*4-2 for _ in range(***number of w's***)]` for a random function
### len_w
The amount of w's that the neural network can use
### batches
The amount of batches the neural network will train on. Set to 0 to get an animation
### batch_size
The amount of points the neural network will train on, before updating the weights
### lr
How fast and good it trains (high = fast, low = good)
### momentum
How much the previous learning direction will affect the new learning direction (weight update)
### points
The amount of randomly generated data points used to learn the polynom
## Graph
- Green: The target function
- Blue: The data points
- Red: The learned function
## Examples
### Overfitting
```
Ws = [2, 1]
len_w = 20
batches = 20000
batch_size = 10
lr = 0.01
momentum = 0.9
points = 10
```
The line wont follow the target function, but get closer to the points
### Too high learning rate
```
Ws = [random.random()*4-2 for _ in range(20)]
len_w = 20
batches = 0
batch_size = 10
lr = 0.1
momentum = 0
points = 200
```
### Too low batch size
```
Ws = [random.random()*4-2 for _ in range(2)]
len_w = 2
batches = 0   #0 for anim
batch_size = 1
lr = 0.01
momentum = 0#.9
points = 20
```
It will only train on one point on a time and move around
### Good learning
```
Ws = [random.random()*4-2 for _ in range(20)]
len_w = 20
batches = 0   #0 for anim
batch_size = 10
lr = 0.01
momentum = 0.9
points = 200
```
Will be pretty good (,but not perfect, because the points aren't evenly distributed)
