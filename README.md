# Decision-Trees

Decision trees are supervised learning algorithms used for both, classification and regression task
Decision trees are assigned to the information based learning algorithms which use different measures of information gain for learning.
We can use decision trees for issues where we have continuous but also categorical input and target features. The main idea of decision trees is to find those descriptive features which contain the most "information" regarding the target feature and
then split the dataset along the values of these features such that the target feature values for the resulting sub_datasets are as pure as possible
It is similar to finding the Principal Components which have a greater effect on the target feature ,we call that component/feature as the most informative one.
This process of finding the "most informative" feature is done until we accomplish a stopping criteria where we then finally end up in so called leaf nodes.
A decision tree mainly contains of a root node, interior nodes, and leaf nodes which are then connected by branches.There can be only one root node and many internal nodes represented by ellipses and the leaf node is denoted by reactangular box.

How do we train the decision tree to make decisions?
1.Present a dataset containing of a number of training instances characterized by a number of descriptive features and a target
  feature
2.Train the decision tree model by continuously splitting the target feature along the values of the descriptive features using a
  measure of information gain during the training process
3.Grow the tree until we accomplish a stopping criteria --> create leaf nodes which represent the predictions we want to make for
  new query instances
4.Show query instances to the tree and run down the tree until we arrive at leaf nodes

We know that this model can make predictions for unknown query instances because it models the relationship between the known descriptive features and the know target feature.

All this is fine,how do we build a tree in the first hand?
What do we want to do with the decision tree? We want, given a dataset, train a model which kind of learns the relationship between the descriptive features and a target feature such that we can present the model a new, unseen set of query instances and predict the target feature values for these query instances.

For example the animals are classified as being Mammals or Reptiles based on whether they are toothed, have legs and do breath.Here the target features are Mammal or Reptile(Species) and the descriptive features are whether they are toothed,have legs and do breath,legs,hair etc., 

Therefore the leaf nodes should contain only Mammal or Reptile.We need to classify the training set so that one set consists of only mammals and other set consists of only reptiles.So,intutively we can say that all that which has hair is a Mammal and which do not is a Reptile.
Our "root node" here is "Hair",if true it is a Mammal,if false it is a Reptile.Imagine a Tree with "Hair" in an ellipse and two branches one for True and other for False.Mammal in rectangular box(leaf node) towards the True branch and Reptile towards the False branch inside a rectangular box.

That is, we have split our dataset by asking the question if the animal has hair or not.Is this enough to classify whether it is a reptile or mammal?We need to further split the data by placing more queries.

We have seen that the Hair feature seems to have more information in splitting the data ie., it has more informativeness.Let us call it as information gain to measure the informativeness.Information Gain(IG) is a measure of how good a descriptive feature is suited to split a dataset on.
To calculate IG we have to introduce the term Entropy of dataset.The Entropy of a dataset is used to measure the impurity of a dataset.There are also many other measurements like Gini Index,Variance,Chi-Square etc.,

So what is Entropy?Suppose you have a bag consisting of 100 white balls(Pure).If you draw a ball from the bag the probability of picking a white ball is 1 and the Entropy is 0.Now we replace 20 white balls with Blue balls and 30 by Red ones(Impure).The probability of picking a white ball now is reduced to 0.5 because the entropy has increased.
Therefore we can say that,the more "impure" a dataset, the higher the entropy and the less "impure" a dataset, the lower the entropy.
Also if we have more than one target feature value, the impurity will be greater than zero.It is useful to sum up the entropies of each possible target feature value and weight it by the probability that we randomly picked up the required class.

				IG= -[(sum of{P(x=k)*log2(P(x=k))}]

Let us find the IG for our example.

			WhiteBall= -(0.5(log2(0.5))= -0.5
			BlueBall= -(0.2(log(0.2))= -0.464
			RedBall= -(0.3(log(0.3))= -0.521
Total IG= -(-0.5-0.464-0.521)= 1.485

Now our task is to find the best feature in terms of information gain (Remember that we want to find the feature which splits the data most accurate along the target feature values) which we should use to first split our data on (which serves as root node)

We find the most informative feature by splitting the dataset using that feature and find the information gain with respective to that feature and subtract it from the Entropy of the dataset to see how much this feature splitting reduces the original entropy.

					IG(feature)= (Entropy(D) - Entropy(feature)
The feature with more IG is the feature that best splits the dataset.

Advantges and Disadvantages of Decision Trees:
Advantages:	Easy to interpret
		Need not normalize the data
		It can handle both continuous and categorical data
		Non linear relations can also be modeled
Disadvantages:	If there are too many continuous features the tree becomes quite large and interpretation may be difficult.
		Decision Trees are prone to OVERFIT the training data.(Solutions will be discussed)
		Small changes in data may lead to form a completely different tree
		If the data has target features that occur frequently in the dataset,the tree may be biased towards that 		 feature.
		
What if the target features are continuously scaled rather than categorically scaled?
Then we call the tree model as Regression Tree model instead of Classification tree model.Therefore we can use the variance of the target feature as splitting measure instead of using the IG.The feature with lowest weighted variance is the best splitting feature

Solution to afore mentioned problem of Overfitting:
One approach to increase the accuracy of a tree model is to use an ensemble approach. With an ensemble approach we create different models (in this case) trees from the original dataset and let the different models make a majority vote on the test dataset. That is, we predict the target values for the test dataset using each of the created models and then return this target feature value which has been predicted by the majority of the models.
The most prominent approaches to create decision tree ensemble models are called bagging and boosting.
A variant of a boosting-based decision tree ensemble model is called random forest model which is one of the most powerful machine learning algorithms.
Bagging:

 Bagging stands for Bootstrap Aggregation.
 Suppose there are N observations and M features. A sample from observation is selected randomly with replacement.
 A subset of features are selected to create a model with sample of observations and subset of features.
 Feature from the subset is selected which gives the best split on the training data.This is repeated to create many models and every model is trained in parallel.
 Prediction is made based on the aggregation of predictions from all the models.
 The only parameters when bagging decision trees is the number of samples and hence the number of trees to include. 
This can be chosen by increasing the number of trees on run after run until the accuracy begins to stop showing improvement.
In the process of bagging we are not concerned about individual trees overfitting the training data.

Boosting:

A bunch of weak learners which performs just slightly better than random guessing can be combined to make better predictions than one strong learner.
Boosting refers to a group of algorithms that utilize weighted averages to make weak learners into stronger learners.
Unlike bagging,boosting is all about “teamwork”.Each model that runs, dictates what features the next model will focus on.After each training step, the weights are redistributed.After each training step, the weights are redistributed.Misclassified data increases its weights to emphasise the most difficult cases.
In this way, subsequent learners will focus on them during their training.

Both are good at reducing the variances and providing higher stability but,only boosting tries to reduce the bias.
Bagging solves the overfitting problem whereas too much of boosting may result in overfitting.
