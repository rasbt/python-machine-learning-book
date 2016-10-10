Sebastian Raschka, 2015

Python Machine Learning - Code Examples

## Chapter 13 - Parallelizing Neural Network Training with Theano

- Building, compiling, and running expressions with Theano
  - What is Theano?
  - First steps with Theano
  - Configuring Theano
  - Working with array structures
  - Wrapping things up – a linear regression example
- Choosing activation functions for feedforward neural networks
  - Logistic function recap
  - Estimating probabilities in multi-class classification via the softmax function
  - Broadening the output spectrum by using a hyperbolic tangent
- Training neural networks efficiently using Keras
- Summary

## Install Theano with g++ under 64-bit Windows

Li-Yi Wei, 2016

I basically follow the instructions from [here](https://www.kaggle.com/c/otto-group-product-classification-challenge/forums/t/13973/a-few-tips-to-install-theano-on-windows-64-bits) and [here](http://rosinality.ncity.net/doku.php?id=python:installing_theano) but without setting paths (as I have already done so).

1. Install MinGW 64bit.
Download mingw-w64-install.exe at [here](http://sourceforge.net/projects/mingw-w64/files/) and install it.
Architecture is x86_64, Threads is posix, Exception is seh.

2. Install MSYS.
Download MSYS-20111123.zip at [here](http://sourceforge.net/projects/mingw-w64/files/External%20binary%20packages%20%28Win64%20hosted%29/MSYS%20%2832-bit%29/) and extract archive under the directory where MinGW installed (directory that has bin subdirectory).
Run msys.bat and type command

    $ sh /postinstall/pi.sh

at the prompt.

3. You should make libpythonXX.a (XX is version number) file manually.
MinGW supports 32bit lib but 64bit not.
First copy pythonXX.dll (i.e. python34.dll) to temporary directory.
Generally you can found pythonXX.dll under C:\Windows\System32 or [Anaconda directory]\libs.
Then run these commands:

    $ gendef pythonXX.dll  
    $ dlltool --as-flags=--64 -m i386:x86-64 -k --output-lib libpythonXX.a --input-def pythonXX.def

4. Copy libpythonXX.a file that you made under the [Python directory]\libs.

5. Make .theanorc under HOME directory with content like this:

    [blas]  
    ldflags = 
    
    [gcc]  
    cxxflags = -shared -I[MinGW directory]\include -L[Python directory]\libs -lpython34 -DMS_WIN64

6. This is my .theanorc:

    [blas]
    ldflags = 
     
    [gcc]
    cxxflags = -shared -I C:\programs\MinGW\mingw64\include -L C:\programs\Anaconda3\libs -lpython35 -DMS_WIN64 -D_hypot=hypot

7. Run this to verifying installation:

    import theano  
    theano.test()

## Install Keras

The default Keras backend seems to be TensorFlow.
See [here](https://keras.io/backend/) about switching to Theano.

