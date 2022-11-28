# Electromyography-and-Gradient-Boosting
## Background and Documentation
### Gradient Boosting
#### History
Graident boosing is very popular and you will most likley see it more often if you are continuing a future in machine learning. The most common recipe for Gradient Boosting is XGBoost which is the standard for winning machine learning competitions. The idea of Gradient Boosting came from can we make a strong model using these simplier/weaker models. Before gradient boosting there was a model called Adaboost which was a greedy algorithm that build a linear combination of simple models by re-weighing the input data. Then, the model (usually a decision tree) is built on earlier incorrectly predicted objects, which are now given larger weights. Adaboost worked miracles but with little explanation at the time to its miracles people though it was just overfitting. This overfitting problem did exist and thats why a few professors in the statistics department at Stanford, who had created Lasso, Elastic Net, and Random Forest, created Gradient Boosting Machine. 

#### How Gradient Boosting Works
Gradient boosting models combines the work of several weaker models o create a strong model.  Gradient boosting optimizes the mean squared error (MSE), also called the L2 loss or cost. The more weaker models used the more the MSE is reduced. To construct a boosted regression model, let's start by creating a crappy model, $f_{0}(x)$ , that predicts an initial approximation of $y$ given feature vector . Then, let's gradually nudge the overall $F_{m}(x)$ model towards the known target value y by adding one or more tweaks, $\Delta_m(x)$. Therefore 
$$\hat{y}= f_{0}(x) + \Delta_1(x) + \Delta_2(x) + \Delta_3(x) +\ ... \ + \Delta_M(x)$$ 
Optimizing a model according to MSE makes it chase outliers because squaring the difference between targets and predicted values emphasizes extreme values. When we can't remove outliers, it's better to optimize the mean absolute error.

#### Gradient Boosting Regression
Gradient Boosting regression would be a similar case 
to Gradient Boosting, however we now involve the learning rate.
$$F_m(X) = F_{m-1}(X) + ηΔ_m(X)$$
therefore:
$$F_1(X) = F_{0} + ηΔ_1(X),  F_2(X) = F_{1} + ηΔ_2(X),$$

#### Modeling Performance
To answer that, we need a loss or cost function, $L(y,\hat{y})$ or $L(y_i,\hat{y}$, that computes the cost of predicting $\hat{y}$ instead of $y$. The loss across all $N$ observations is just the average (or the sum if you want since $N$ is a constant once we start training) of all the individual observation losses:
$$L(y, F_M(X)) = \frac{1}{N}Σ^{N}_{i = 1}L(y_i- F_M(x_i))$$
The mean squared error (MSE) is the most common:
$$L(y, F_M(X)) = \frac{1}{N}Σ^{N}_{i = 1}(y_i- F_M(x_i))^2$$

#### Hyper-parameters
The two parameters we discuss are the number of stages M and the learning rate . Both affect model accuracy. The more stages we use, the more accurate the model, but the more likely we are to be overfitting. The primary value of the learning rate is to reduce overfitting of the overall model. As a side note, the idea of using a learning rate to reduce overfitting in models that optimize cost functions to learn, such as deep learning neural networks, is very common. Rather than using a constant learning rate, though, we could start the learning rate out energetically and gradually slow it down as the model approached optimality. For our experiments we stuck to a fixed parameters:
$$M = 200, η = 1$$ (which is the default learning rate for the decision tree we are using.)
