# Building a Shallow Neural Network using `numpy`

In the previous notebook, we looked at a general overview of using gradient descent to solve for the weights and biases of a shallow neural network with non-linear activations. We'll now formalize our gradient descent procedure, and then code it up using `numpy`. 

## Gradient Descent for a Shallow Neural Network

### Gradient Descent Procedure 

Our gradient descent procedure is going to involve the same steps that it did in linear and logistic regression. Formally, we'll take the following steps:

1. Randomly initialize values for our weights and biases: 
<img src="../imgs/variables/w2_11.png" style="vertical-align: text-middle; display: inline-block; padding-top:0; margin-top:0;" width=32 \>,
<img src="../imgs/variables/w2_12.png" style="vertical-align: text-middle; display: inline-block; padding-top:0; margin-top:0;" width=32 \>,
<img src="../imgs/variables/w3_11.png" style="vertical-align: text-middle; display: inline-block; padding-top:0; margin-top:0;" width=31 \>,
<img src="../imgs/variables/w3_21.png" style="vertical-align: text-middle; display: inline-block; padding-top:0; margin-top:0;" width=31 \>,
<img src="../imgs/variables/b2_1.png" style="vertical-align: text-middle; display: inline-block; padding-top:0; margin-top:0;" width=16 \>,
<img src="../imgs/variables/b2_2.png" style="vertical-align: text-middle; display: inline-block; padding-top:0; margin-top:0;" width=20 \>,
and 
<img src="../imgs/variables/b3_1.png" style="vertical-align: text-middle; display: inline-block; padding-top:0; margin-top:0;" width=19\>. Weights are often initialized following a uniform or normal distribution, whereas biases are frequently initialized with zeros. 

2. While we haven't met some stopping condition: 

 A. Calculate our predicted values, 
<img src="../imgs/variables/yhat.png" style="vertical-align: text-middle; display: inline-block; padding-top:0; margin-top:0;" width=13 \>.    
 B. Calculate the error for each observation using the true values
<img src="../imgs/variables/yi.png" style="vertical-align: text-middle; display: inline-block; padding-top:0; margin-top:0;" width=17 \>, 
our predicted values 
<img src="../imgs/variables/yhati.png" style="vertical-align: text-middle; display: inline-block; padding-top:0; margin-top:0;" width=16 \>, 
and our error formula: 
<img src="../imgs/equations/ind_squared_error.png" width=110 \>
 C. For each observation, calculate the gradient of the error with respect to each one of our weights and biases (<img src="../imgs/derivatives/ei_w2_11.png" style="vertical-align: text-middle; display: inline-block; padding-top:0; margin-top:0;" width=42 \>, 
<img src="../imgs/derivatives/ei_w2_12.png" style="vertical-align: text-middle; display: inline-block; padding-top:0; margin-top:0;" width=42 \>, 
<img src="../imgs/derivatives/ei_w3_11.png" style="vertical-align: text-middle; display: inline-block; padding-top:0; margin-top:0;" width=40 \>, 
<img src="../imgs/derivatives/ei_w3_21.png" style="vertical-align: text-middle; display: inline-block; padding-top:0; margin-top:0;" width=40 \>, 
<img src="../imgs/derivatives/ei_b2_1.png" style="vertical-align: text-middle; display: inline-block; padding-top:0; margin-top:0;" width=32 \>, 
<img src="../imgs/derivatives/ei_b2_2.png" style="vertical-align: text-middle; display: inline-block; padding-top:0; margin-top:0;" width=32 \>, 
and 
<img src="../imgs/derivatives/ei_b3_1.png" style="vertical-align: text-middle; display: inline-block; padding-top:0; margin-top:0;" width=32 \>), and then use the average across observations to update (<img src="../imgs/variables/alpha.png" style="vertical-align: text-middle; display: inline-block; padding-top:0; margin-top:0;" width=15\>
 is the learning rate):
 
<img src="../imgs/updates/w2_11_update.png" width=175 \> 
<img src="../imgs/updates/w2_12_update.png"  width=175 \> 
<img src="../imgs/updates/w3_11_update.png"  width=175 \> 
<img src="../imgs/updates/w3_21_update.png"  width=175 \> 
<img src="../imgs/updates/b2_1_update.png"   width=150 \> 
<img src="../imgs/updates/b2_2_update.png"   width=150 \> 
<img src="../imgs/updates/b3_1_update.png"   width=150 \>

### Derivative Calculations

We'll revisit the chain rule that we looked at last notebook in order to calculate the gradients for all the observations in 2C:

<img src="../imgs/derivatives/ei_w2_11_chain.png" width=225 \>
<img src="../imgs/derivatives/ei_w2_12_chain.png" width=225 \> 
<img src="../imgs/derivatives/ei_w3_11_chain.png" width=130 \> 
<img src="../imgs/derivatives/ei_w3_21_chain.png" width=130 \> 
<img src="../imgs/derivatives/ei_b2_1_chain.png"  width=210 \> 
<img src="../imgs/derivatives/ei_b2_2_chain.png"  width=210 \> 
<img src="../imgs/derivatives/ei_b3_1_chain.png"  width=110 \>  

We can break these equations down into the individual pieces that we need to calculate - 
<img src="../imgs/derivatives/ei_yi.png" style="vertical-align: text-middle; display: inline-block; padding-top:0; margin-top:0" width=30\>, 
<img src="../imgs/derivatives/yhati_sigma_z2_1i.png" style="vertical-align: text-middle; display: inline-block; padding-top:0; margin-top:0" width=60\>,
<img src="../imgs/derivatives/yhati_sigma_z2_2i.png" style="vertical-align: text-middle; display: inline-block; padding-top:0; margin-top:0" width=61\>,
<img src="../imgs/derivatives/yhati_w3_11.png" style="vertical-align: text-middle; display: inline-block; padding-top:0; margin-top:0" width=48\>,
<img src="../imgs/derivatives/yhati_w3_21.png" style="vertical-align: text-middle; display: inline-block; padding-top:0; margin-top:0" width=41\>,
<img src="../imgs/derivatives/yhati_b3_1.png" style="vertical-align: text-middle; display: inline-block; padding-top:0; margin-top:0" width=37\>,
<img src="../imgs/derivatives/sigma_z2_1i_z2_1i.png" style="vertical-align: text-middle; display: inline-block; padding-top:0; margin-top:0" width=61\>,
<img src="../imgs/derivatives/sigma_z2_2i_z2_2i.png" style="vertical-align: text-middle; display: inline-block; padding-top:0; margin-top:0" width=61\>,
<img src="../imgs/derivatives/z2_1i_w2_11.png" style="vertical-align: text-middle; display: inline-block; padding-top:0; margin-top:0" width=46\>,
<img src="../imgs/derivatives/z2_2i_w2_12.png" style="vertical-align: text-middle; display: inline-block; padding-top:0; margin-top:0" width=41\>,
<img src="../imgs/derivatives/z2_1i_b2_1.png" style="vertical-align: text-middle; display: inline-block; padding-top:0; margin-top:0" width=35\>,
<img src="../imgs/derivatives/z2_2i_b2_2.png" style="vertical-align: text-middle; display: inline-block; padding-top:0; margin-top:0" width=38\> - and we can calculate these as follows: 

<img src="../imgs/derivatives/ei_yi_soln.png" width=275 \>
<img src="../imgs/derivatives/yhati_sigma_z2_1i_soln.png" width=125 \>
<img src="../imgs/derivatives/yhati_sigma_z2_2i_soln.png" width=125 \>
<img src="../imgs/derivatives/yhati_w3_11_soln.png" width=125 \>
<img src="../imgs/derivatives/yhati_w3_21_soln.png" width=125 \>
<img src="../imgs/derivatives/yhati_b3_1_soln.png" width=70 \>
<img src="../imgs/derivatives/sigma_z2_1i_z2_1i_soln.png" width=260 \>
<img src="../imgs/derivatives/sigma_z2_2i_z2_2i_soln.png" width=260 \>
<img src="../imgs/derivatives/z2_1i_w2_11_soln.png" width=90 \>
<img src="../imgs/derivatives/z2_2i_w2_12_soln.png" width=90 \>
<img src="../imgs/derivatives/z2_1i_b2_1_soln.png" width=80 \>
<img src="../imgs/derivatives/z2_2i_b2_2_soln.png" width=80 \>

Plugging these back into the original equations, we can obtain our full updates for step 2C:


<img src="../imgs/derivatives/ei_w2_11_chain_soln.png" width=450 \>
<img src="../imgs/derivatives/ei_w2_12_chain_soln.png" width=430 \> 
<img src="../imgs/derivatives/ei_w3_11_chain_soln.png" width=230 \> 
<img src="../imgs/derivatives/ei_w3_21_chain_soln.png" width=230 \> 
<img src="../imgs/derivatives/ei_b2_1_chain_soln.png"  width=430 \> 
<img src="../imgs/derivatives/ei_b2_2_chain_soln.png"  width=420 \> 
<img src="../imgs/derivatives/ei_b3_1_chain_soln.png"  width=250 \>

Alright, enough math for now. Let's dive into some code!

## Learning A Shallow Neural Network using Gradient Descent with `numpy`

To learn the weights and biases of our shallow neural-network, we'll first need some data. There are two functions in `datasets.general` that we'll use to generate some fake data, and then we'll demonstrate that our neural network is capable of learning the functions that generated this data. 

For the most part, the code below is going to resemble the equations that we've looked at so far. Probably the biggest difference is that we'll create weight and bias vectors per layer, rather than creating individual variables for each weight and bias (see the [appendix](https://github.com/sallamander/neural-networks-intro/blob/master/mini-books/shallow-neural-networks/04-shallow-neural-network/appendix.ipynb) for a non-vector version of the code). This will make the code a little cleaner and a little faster. It's also how standard neural network libraries operate, and it'll be good to start getting used to it while the networks we're building are relatively simple.  