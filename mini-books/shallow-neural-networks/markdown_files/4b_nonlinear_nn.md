# A Shallow Neural Network with Non-Linear Activations

Last notebook, we looked at a feedforward, fully-connected neural network with one hidden layer where each node's activation was the identity function. We discovered that this was really just simple linear regression, and suggested that the real power of a neural network would come with the use of non-linear activation functions. We'll now examine that claim by using the sigmoid activation that we used during logistic regression...

## A Neural Network with Sigmoid Activations

If we take our [computational graph](../imgs/custom/shallow_linear_connect.png) from last notebook and simply replace the activations in the hidden layer with sigmoid activations, our computational graph becomes the following:

<img src="../imgs/custom/shallow_nonlinear_connect.png" width=450 \>

and our mathematical depiction of this neural network becomes:

<img src="../imgs/equations/shallow_nonlinear_formula1.png" width=250 \>

As we've done with linear and logistic regression, we then solve for the weights through gradient descent. We'll use **squared error** as our error metric, as is fitting since we are using an identity activation function in the output layer. Thus, the error for an *individual* observation is: 

 <img src="../imgs/equations/ind_squared_error.png" width=115 \>

 and the average error across *all* observations is: 

 <img src="../imgs/equations/agg_squared_error.png" width=150 \>
 
 
#### Notation Changes
 
Before diving into gradient descent, we'll make one notational change. Recall that in the [derivative calculations](https://github.com/sallamander/neural-networks-intro/blob/master/mini-books/shallow-neural-networks/03-logistic/3b_nn_np.ipynb) for logistic regression, we denoted the weighted sum of the inputs to a node using a 
<img src="../imgs/variables/z.png" style="vertical-align: text-middle; display: inline-block; padding-top:0; margin-top:0;" width=17 \>, which was done to make the calculations a little easier to work with. We'll do the same here, but we have to factor in the fact that there are multiple nodes receiving inputs. To account for this, we'll use a *subscript* as well as a *superscript* with each 
<img src="../imgs/variables/z.png" style="vertical-align: text-middle; display: inline-block; padding-top:0; margin-top:0;" width=17 \>. 
Thus, 
<img src="../imgs/variables/z_j_l.png" style="vertical-align: text-middle; display: inline-block; padding-top:0; margin-top:0;" width=24 \>
will denote the weighted sum of the inputs to the 
<img src="../imgs/variables/jth.png" style="vertical-align: text-middle; display: inline-block; padding-top:0; margin-top:0;" width=22 \>
node in the 
<img src="../imgs/variables/lth.png" style="vertical-align: text-middle; display: inline-block; padding-top:0; margin-top:0;" width=22 \>
layer. If we apply this change to our mathematical equation above, we can denote the inputs to each node in Layer 2 much more compactly:

 <img src="../imgs/equations/z2_1.png" width=150 \>
 
 <img src="../imgs/equations/z2_2.png" width=150 \>

which allows us to denote the mathematical depiction of our neural network a little more concisely:

<img src="../imgs/equations/shallow_nonlinear_formula2.png" width=250 \> 
   
**Note**: Technically, 
<img src="../imgs/equations/y_z1_3.png" style="vertical-align: text-middle; display: inline-block; padding-top:0; margin-top:0;" width=55 \>
, but we didn't make that change so that we'll be able to more easily follow along with how our predictions for 
<img src="../imgs/variables/y.png" style="vertical-align: text-middle; display: inline-block; padding-top:0; margin-top:0;" width=13 \>
are generated. 

### Solving For the Weights Via Gradient Descent 

Solving for the values of the weights and biases in our neural network can be done using gradient descent. Similar to using gradient descent with linear and logistic regression, we'll have a forward propagation and a backward propagation step. Before formalizing the gradient descent procedure in the next notebook, let's visualize it in a computational graph. 

### Forward Propagation

With forward propagation, we'll simply read our computational graph from **left to right** in order to compute our predicted values (
<img src="../imgs/variables/yhat.png" style="vertical-align: text-middle; display: inline-block; padding-top:0; margin-top:0;" width=13 \>). 

<img src="../imgs/custom/shallow_nonlinear_connect_forprop.png" width=450 \>

### Backward Propagation

With backward propagation, we'll read our computational graph in the opposite direction, from **right to left**. The inputs that we start out with will be our errors that we calculate for each observation. In backward propagation, we'll calculate the gradient of each of the errors with respect to each of the weights and biases - 
<img src="../imgs/derivatives/ei_w2_11.png" style="vertical-align: text-middle; display: inline-block; padding-top:0; margin-top:0;" width=42 \>, 
<img src="../imgs/derivatives/ei_w2_12.png" style="vertical-align: text-middle; display: inline-block; padding-top:0; margin-top:0;" width=42 \>, 
<img src="../imgs/derivatives/ei_w3_11.png" style="vertical-align: text-middle; display: inline-block; padding-top:0; margin-top:0;" width=40 \>, 
<img src="../imgs/derivatives/ei_w3_21.png" style="vertical-align: text-middle; display: inline-block; padding-top:0; margin-top:0;" width=40 \>, 
<img src="../imgs/derivatives/ei_b2_1.png" style="vertical-align: text-middle; display: inline-block; padding-top:0; margin-top:0;" width=32 \>, 
<img src="../imgs/derivatives/ei_b2_2.png" style="vertical-align: text-middle; display: inline-block; padding-top:0; margin-top:0;" width=32 \>, 
and 
<img src="../imgs/derivatives/ei_b3_1.png" style="vertical-align: text-middle; display: inline-block; padding-top:0; margin-top:0;" width=32 \>. 

<br>

As before, we'll obtain these using the chain rule:

<img src="../imgs/derivatives/ei_w2_11_chain.png" width=225 \>
<img src="../imgs/derivatives/ei_w2_12_chain.png" width=225 \> 
<img src="../imgs/derivatives/ei_w3_11_chain.png" width=130 \> 
<img src="../imgs/derivatives/ei_w3_21_chain.png" width=130 \> 
<img src="../imgs/derivatives/ei_b2_1_chain.png"  width=210 \> 
<img src="../imgs/derivatives/ei_b2_2_chain.png"  width=210 \> 
<img src="../imgs/derivatives/ei_b3_1_chain.png"  width=110 \>  

<br>
Visualizing this through our computational graph would look as follows: 

<img src="../imgs/custom/shallow_nonlinear_connect_backprop.png" width=600 \>

Phew, this is a lot of derivatives to calculate! In the next notebook, we'll walk through the calculations of these derivatives, and then code up this shallow neural network using `numpy`. 