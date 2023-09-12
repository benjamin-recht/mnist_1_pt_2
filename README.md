# mnist_1_pt_2

I'm not sure why anyone cares, but here's some code to get 1.18% accuracy on MNIST using only least squares and numpy calls.

You can get the MNIST data set [here](https://s3.amazonaws.com/img-datasets/mnist.npz.)

The algorithm computes the minimum norm solution

f(x_i) = y_i

where y_i is a one-hot encoding of the MNIST training labels. The feature space is that of the quadratic kernel

k(x,z) = <x/norm(x),z/norm(z)>^4

The code is less than 10 lines of python. I'm sure I could code-golf this to less. Whatever.

I also added a version with numba which is much faster. numpy's component-wise matrix operations seem to still be serial and slow.
