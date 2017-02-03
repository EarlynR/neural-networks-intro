# Multiple Linear Regression using Gradient Descent

In the last notebook, we walked through a high level overview of using gradient descent to solve our multiple linear regression problem. After building in a little more detail and calculating the derivatives that we'll need to perform the update steps, we'll code it up in `numpy`. 

## Gradient Descent for Multiple Linear Regression

### Gradient Descent Procedure 

With gradient descent, we'll do the following: 

1. Randomly initialize values for our coefficients: 
<img src="../imgs/variables/beta0.png" style="vertical-align: text-middle; display: inline-block; padding-top:0; margin-top:0;" width=20 \>, 
<img src="../imgs/variables/beta1.png" style="vertical-align: text-middle; display: inline-block; padding-top:0; margin-top:0;" width=22 \> , 
<img src="../imgs/variables/beta2.png" style="vertical-align: text-middle; display: inline-block; padding-top:0; margin-top:0;" width=19 \> , and
<img src="../imgs/variables/beta3.png" style="vertical-align: text-middle; display: inline-block; padding-top:0; margin-top:0;" width=18 \>. 

2. While we haven't met some stopping condition:   
 A. Calculate our predicted values, 
<img src="../imgs/variables/yhat.png" style="vertical-align: text-middle; display: inline-block; padding-top:0; margin-top:0;" width=13 \>.  
 B. Calculate the error for each observation using the true values
<img src="../imgs/variables/yi.png" style="vertical-align: text-middle; display: inline-block; padding-top:0; margin-top:0;" width=17 \>, 
our predicted values 
<img src="../imgs/variables/yhati.png" style="vertical-align: text-middle; display: inline-block; padding-top:0; margin-top:0;" width=16 \>, 
and our error formula: 
<img src="../imgs/equations/ind_squared_error.png" width=110 \>      
 C. For each observation, calculate the gradient of the error with respect to each one of our coefficients (
<img src="../imgs/derivatives/ei_beta0.png" style="vertical-align: text-middle; display: inline-block; padding-top:0; margin-top:0;" width=30\>, 
<img src="../imgs/derivatives/ei_beta1.png" style="vertical-align: text-middle; display: inline-block; padding-top:0; margin-top:0;" width=30\>, 
<img src="../imgs/derivatives/ei_beta2.png" style="vertical-align: text-middle; display: inline-block; padding-top:0; margin-top:0;" width=30\>, 
<img src="../imgs/derivatives/ei_beta3.png" style="vertical-align: text-middle; display: inline-block; padding-top:0; margin-top:0;" width=30\>
), and then use the average across observations to update the coefficients (
<img src="../imgs/variables/alpha.png" style="vertical-align: text-middle; display: inline-block; padding-top:0; margin-top:0;" width=15\>
 is the learning rate): 
<img src="../imgs/updates/beta0_simp_linear_update.png" width=150 \>
<img src="../imgs/updates/beta1_simp_linear_update.png" width=150 \>
<img src="../imgs/updates/beta2_simp_linear_update.png" width=150 \>
<img src="../imgs/updates/beta3_simp_linear_update.png" width=150 \>

### Derivative Calculations

To calculate the gradients for each observation in 2C, we'll use the chain rule that we looked at last notebook: 

<img src="../imgs/derivatives/ei_beta0_chain.png" width=120\>
<img src="../imgs/derivatives/ei_beta1_chain.png" width=120\>
<img src="../imgs/derivatives/ei_beta2_chain.png" width=120\>
<img src="../imgs/derivatives/ei_beta3_chain.png" width=120\>

We can break these equations down into calculating each of the individual pieces - 
<img src="../imgs/derivatives/ei_yi.png" style="vertical-align: text-middle; display: inline-block; padding-top:0; margin-top:0" width=30\>, 
<img src="../imgs/derivatives/yhati_beta0.png" style="vertical-align: text-middle; display: inline-block; padding-top:0; margin-top:0" width=30\>, 
<img src="../imgs/derivatives/yhati_beta1.png" style="vertical-align: text-middle; display: inline-block; padding-top:0; margin-top:0" width=30\>
<img src="../imgs/derivatives/yhati_beta2.png" style="vertical-align: text-middle; display: inline-block; padding-top:0; margin-top:0" width=30\>, 
<img src="../imgs/derivatives/yhati_beta3.png" style="vertical-align: text-middle; display: inline-block; padding-top:0; margin-top:0" width=35\>. We can calculate those as follows: 

<img src="../imgs/derivatives/ei_yi_soln.png" width=275 \>
<img src="../imgs/derivatives/yhati_beta0_soln.png" width=75 \>
<img src="../imgs/derivatives/yhati_beta1_soln.png" width=90 \>
<img src="../imgs/derivatives/yhati_beta2_soln.png" width=90 \>
<img src="../imgs/derivatives/yhati_beta3_soln.png" width=90 \>

If we plug these back into the original equations, we can obtain our full updates for step 2C: 

<img src="../imgs/derivatives/ei_beta0_chain_soln.png" width=350\>
<img src="../imgs/derivatives/ei_beta1_chain_soln.png" width=290\>
<img src="../imgs/derivatives/ei_beta2_chain_soln.png" width=290\>
<img src="../imgs/derivatives/ei_beta3_chain_soln.png" width=290\>

Now, let's code this up! 

## Multiple Linear Regression using Gradient Descent with `numpy`

To demonstrate using gradient descent to solve our multiple linear regression problem, we'll use the `gen_multiple_linear` function from the `datasets/general.py` script to generate some toy data that follows a multivariate linear relationship with three variables. We'll input a `1d numpy array` of betas as well as a number of observations, and it will output data that follows a multivariate linear relationship (
<img src="../imgs/equations/mult_linear_3_feats.png" width=120 style="vertical-align: text-middle; display: inline-block; padding-top:0; margin-top:0;" \> ). 
With this data, we'll use gradient descent to learn the values for our coefficients. 

In solving our multiple linear regression problem, we'll work exclusively with vectors and matrices. Instead of having individual beta coefficients (like we did with `beta_0` and `beta_1` in simple linear regression), we'll have a beta vector that will hold each of our betas. This means that the first column of the `xs` matrix returned from `gen_multiple_linear` will be a vector of 1's that will be lined up with our 
<img src="../imgs/variables/beta0.png" style="vertical-align: text-middle; display: inline-block; padding-top:0; margin-top:0;" width=20 \>. Aside from this, our solution for multiple linear regression will look largely the same as our solution for simple linear regression. 