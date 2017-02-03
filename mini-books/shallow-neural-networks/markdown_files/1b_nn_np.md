# Simple Linear Regression using Gradient Descent

In the previous notebook, we looked at a high level overview of solving our simple linear regression problem using gradient descent. Now, we'll formalize that, actually calculate the derivatives that we need to implement it, and use `numpy` to do so. 

## Using Gradient Descent for Simple Linear Regression

### Gradient Descent Procedure 

Formally, with gradient descent we will do the following: 

1. Randomly initialize values for our coefficients, 
<img src="../imgs/variables/beta0.png" style="vertical-align: text-middle; display: inline-block; padding-top:0; margin-top:0;" width=20 \> and
<img src="../imgs/variables/beta1.png" style="vertical-align: text-middle; display: inline-block; padding-top:0; margin-top:0;" width=22 \>   

2. While we haven't met some stopping condition:   
 A. Calculate our predicted values, 
<img src="../imgs/variables/yhat.png" style="vertical-align: text-middle; display: inline-block; padding-top:0; margin-top:0;" width=13 \>, using our simple linear regression equation
(<img src="../imgs/equations/simp_linear.png" width=100 style="vertical-align: text-middle; display: inline-block; padding-top:0; margin-top:0;" \>).  
 B. Calculate the error for each observation using the true values
<img src="../imgs/variables/yi.png" style="vertical-align: text-middle; display: inline-block; padding-top:0; margin-top:0;" width=17 \>, 
our predicted values 
<img src="../imgs/variables/yhati.png" style="vertical-align: text-middle; display: inline-block; padding-top:0; margin-top:0;" width=16 \>, 
and our error formula: 
<img src="../imgs/equations/ind_squared_error.png" width=110 \>      
 C. For each observation, calculate the gradient of the error with respect to each one of our coefficients (
<img src="../imgs/derivatives/ei_beta0.png" style="vertical-align: text-middle; display: inline-block; padding-top:0; margin-top:0;" width=30\>, 
<img src="../imgs/derivatives/ei_beta1.png" style="vertical-align: text-middle; display: inline-block; padding-top:0; margin-top:0;" width=30\>
), and then use the average across observations to update the coefficients: 
<img src="../imgs/updates/beta0_simp_linear_update.png" width=150 \>
<img src="../imgs/updates/beta1_simp_linear_update.png" width=150 \>

Note that we are subtracting off the gradient in step 2C because the gradient gives us the direction of steepest ascent (and we want to take the direction of steepest descent to minimize our error). Note also that 
<img src="../imgs/variables/alpha.png" style="vertical-align: text-middle; display: inline-block; padding-top:0; margin-top:0;" width=15\> 
in step 2C is simply the learning rate (e.g. how much of the coefficient updates actually get applied). 

Now, let's actually calculate the derivatives. 

### Derivative Calculations

Recall that we'll use the chain rule to calculate the updates that we need for step 2C: 

<img src="../imgs/derivatives/ei_beta0_chain.png" width=120\>
<img src="../imgs/derivatives/ei_beta1_chain.png" width=110\>

To calculate these, we'll need three quantities - 
<img src="../imgs/derivatives/ei_yi.png" style="vertical-align: text-middle; display: inline-block; padding-top:0; margin-top:0" width=30\>, 
<img src="../imgs/derivatives/yhati_beta0.png" style="vertical-align: text-middle; display: inline-block; padding-top:0; margin-top:0" width=30\>, 
<img src="../imgs/derivatives/yhati_beta1.png" style="vertical-align: text-middle; display: inline-block; padding-top:0; margin-top:0" width=30\>. We can calculate these as follows: 

<img src="../imgs/derivatives/ei_yi_soln.png" width=275 \>
<img src="../imgs/derivatives/yhati_beta0_soln.png" width=75 \>
<img src="../imgs/derivatives/yhati_beta1_soln.png" width=90 \>

We can then plug these in to obtain our updates for step 2C: 

<img src="../imgs/derivatives/ei_beta0_chain_soln.png" width=350\>
<img src="../imgs/derivatives/ei_beta1_chain_soln.png" width=290\>

Check out these [notes from Andrej Karpathy](http://karpathy.github.io/neuralnets/) for a refresher on gradient descent or a more thorough but still simplistic discussion of backpropagation (the code in the notes is written in `JavaScript`, but it is fairly simplistic and Andrej does an excellent job with his explanations). 

## Simple Linear Regression using Gradient Descent with `numpy`

We'll begin our `numpy` implementation by generating some fake data. To obtain some fake data that follows a univariate linear relationship, we'll use a function from the `datasets/general.py`. With the function `gen_simple_linear`, we'll input a <img src="../imgs/variables/beta0.png" style="vertical-align: text-middle; display: inline-block; padding-top:0; margin-top:0;" width=20 \>, 
<img src="../imgs/variables/beta1.png" style="vertical-align: text-middle; display: inline-block; padding-top:0; margin-top:0;" width=22 \>, 
and number of observations. We'll receive as output data that follows a univariate linear relationship specified by 
 <img src="../imgs/variables/beta0.png" style="vertical-align: text-middle; display: inline-block; padding-top:0; margin-top:0;" width=20 \> 
and 
<img src="../imgs/variables/beta1.png" style="vertical-align: text-middle; display: inline-block; padding-top:0; margin-top:0;" width=22 \> 
(<img src="../imgs/equations/simp_linear.png" style="vertical-align: text-middle; display: inline-block; padding-top:0; margin-top:0;" width=110 \>). Using this data, we'll learn the coefficients using gradient descent as specified above. We'll plot the mean-squared-error over each iteration so that we can see our model learning. 