# Logistic Regression Using Gradient Descent

In the previous notebook, we looked at a high level overview of using gradient descent to solve for the beta coefficients in logistic regression. Now, we'll formalize the gradient descent procedure a little bit more, walk through the calculation of the needed derivatives, and code it all up using `numpy`. 

## Gradient Descent for Logistic Regression 

Before diving into gradient descent, we'll introduce one new piece of notation (this will help with the derivative calculations below) - we're going to denote the weighted sum of the inputs (<img src="../imgs/variables/x_beta.png" style="vertical-align: text-middle; display: inline-block; padding-top:0; margin-top:0;" width=30 \>) as <img src="../imgs/variables/z.png" style="vertical-align: text-middle; display: inline-block; padding-top:0; margin-top:0;" width=17 \>. This is common notation for the **weighted sum of the inputs to a node**, and it will come into play with the derivative calculations in step 2C below. 

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
<img src="../imgs/equations/ind_bin_crossentropy.png" width=350 \>    
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

Before calculating any derivatives, recall that our predicted values, 
<img src="../imgs/variables/yhat.png" style="vertical-align: text-middle; display: inline-block; padding-top:0; margin-top:0;" width=13 \>
, are given by our logistic regression equation: 
<img src="../imgs/equations/logistic_activation2.png" align="center" width="200"\> 
Recall also that we are now replacing 
<img src="../imgs/variables/x_beta.png" style="vertical-align: text-middle; display: inline-block; padding-top:0; margin-top:0;" width=30 \> 
with 
<img src="../imgs/variables/z.png" style="vertical-align: text-middle; display: inline-block; padding-top:0; margin-top:0;" width=17 \>, which means we can denote our logistic regression equation via: 

<img src="../imgs/equations/logistic_activation3.png" align="center" width="175"\>

To calculate the derivatives for an individual observation, we'll use the chain rule that we looked at last notebook, but we'll also have to factor in the non-linear activation function we're now applying (e.g. the sigmoid activation). To denote this, we'll break apart the derivatives of the output with respect to the coefficients (<img src="../imgs/derivatives/yhati_beta0.png" style="vertical-align: text-middle; display: inline-block; padding-top:0; margin-top:0" width=30\>, 
<img src="../imgs/derivatives/yhati_beta1.png" style="vertical-align: text-middle; display: inline-block; padding-top:0; margin-top:0" width=30\>
<img src="../imgs/derivatives/yhati_beta2.png" style="vertical-align: text-middle; display: inline-block; padding-top:0; margin-top:0" width=30\>, 
<img src="../imgs/derivatives/yhati_beta3.png" style="vertical-align: text-middle; display: inline-block; padding-top:0; margin-top:0" width=35\>), and denote 
<img src="../imgs/variables/yhati.png" style="vertical-align: text-middle; display: inline-block; padding-top:0; margin-top:0;" width=16 \>
as <img src="../imgs/variables/sigma_zi.png" style="vertical-align: text-middle; display: inline-block; padding-top:0; margin-top:0;" width=45 \>:

<img src="../imgs/derivatives/ei_beta0_chain_logistic.png" width=300\>
<img src="../imgs/derivatives/ei_beta1_chain_logistic.png" width=300\>
<img src="../imgs/derivatives/ei_beta2_chain_logistic.png" width=300\>
<img src="../imgs/derivatives/ei_beta3_chain_logistic.png" width=300\>

The first two pieces of each of the above chain rules are what we looked at last notebook, whereas the third (rightmost) piece is where we've broken apart the derivatives of the output with respect to the coefficients. For each of these rightmost pieces, we can break them down even farther into each of the individual pieces - 
<img src="../imgs/derivatives/ei_sigma_zi.png" style="vertical-align: text-middle; display: inline-block; padding-top:0; margin-top:0" width=60\>, 
<img src="../imgs/derivatives/sigma_zi_zi.png" style="vertical-align: text-middle; display: inline-block; padding-top:0; margin-top:0" width=50\>, 
<img src="../imgs/derivatives/zi_beta0.png" style="vertical-align: text-middle; display: inline-block; padding-top:0; margin-top:0" width=30\>, 
<img src="../imgs/derivatives/zi_beta1.png" style="vertical-align: text-middle; display: inline-block; padding-top:0; margin-top:0" width=30\>, 
<img src="../imgs/derivatives/zi_beta2.png" style="vertical-align: text-middle; display: inline-block; padding-top:0; margin-top:0" width=32\>, 
<img src="../imgs/derivatives/zi_beta3.png" style="vertical-align: text-middle; display: inline-block; padding-top:0; margin-top:0" width=32\>. We can calculate those as follows: 

<img src="../imgs/derivatives/ei_sigma_zi_soln.png" width=350\> 
<img src="../imgs/derivatives/sigma_zi_zi_soln.png" width=250\> 
<img src="../imgs/derivatives/zi_beta0_soln.png" width=80\>
<img src="../imgs/derivatives/zi_beta1_soln.png" width=100\>
<img src="../imgs/derivatives/zi_beta2_soln.png" width=100\> 
<img src="../imgs/derivatives/zi_beta3_soln.png" width=100\>

If we plug these back into the original equations, we can obtain our full updates for step 2C: 

<img src="../imgs/derivatives/ei_beta0_chain_logistic_soln.png" width=350\>
<img src="../imgs/derivatives/ei_beta1_chain_logistic_soln.png" width=350\>
<img src="../imgs/derivatives/ei_beta2_chain_logistic_soln.png" width=350\>
<img src="../imgs/derivatives/ei_beta3_chain_logistic_soln.png" width=350\>

Note that in the final step of each of the update steps, we have switched from <img src="../imgs/variables/sigma_zi.png" style="vertical-align: text-middle; display: inline-block; padding-top:0; margin-top:0;" width=45 \> 
back to 
<img src="../imgs/variables/yhati.png" style="vertical-align: text-middle; display: inline-block; padding-top:0; margin-top:0;" width=16 \>. This was so that we could see that these are the exact same update steps that we used in solving multiple linear regression with gradient descent! It turns out that when we use a **sigmoid activation function** in combination with **binary crossentropy loss**, we obtain the same update steps as when we use an **identity activation** in combination with **squared error loss**. This means that when we code this up with `numpy` below, the only difference from our `numpy` implementation for multiple linear regression will be the forward pass where we calculate our predicted values. The backward pass will look exactly the same! 

## Logistic Regression using Gradient Descent with `numpy`

To demonstrate using gradient descent to solve logistic regression, we'll use the `gen_multiple_logistic` function from the `datasets/general.py` script to generate some toy data that follows a multivariate logistic relationship with three variables. We'll input a `1d numpy array` of betas as well as a number of observations, and it will output data that follows a multivariate logistic relationship (
<img src="../imgs/equations/logistic.png" width=120 style="vertical-align: text-middle; display: inline-block; padding-top:0; margin-top:0;" \> ). We'll then binarize this data by labeling all those observations greater than 0.5 with a `1`, and all those less than or equal to 0.5 with a `0`.