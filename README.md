# Advanced Data Science Practice
## 1. Overseas merchandise trade
[Kobe Bryant Shot Analysis & Auckland Temperature Study.html](https://isabella051.github.io/Advanced-Data-Science-Practice
/A1.html)
- Write the **Unix** command to create a directory, show information about all the data files and the number of lines in the file.
- Write a shell command that extracts all records with HSC 8703231915 from all the files into a new file car-imports.data.
- Write a shell script that returns total counts for all HS codes in ascending count order and print the last five lines of the output.
- Write one or more shell commands that create a valid CSV file.
- Create a bar plot of total imported value (VFD) by country.
- Compute the aggregated monthly import value (VFD) of each HS code and draw a line plot of the result. (**xtabs**, **matplot**)
- Generate training and test sets.
- Fit two models to the training data: a simple overall mean and a linear regression model with the Date as a predictor variable.
- Calculate RMSE and plot the model predictions.

## 2. Euro foreign currency exchange
- Use the curl command in the Unix shell to download the exchange rate.
- Read the **json** file into R.
- Plot a line plot of all the exchange rates over time.
- Use **pairs** to show pairwise scatterplots between all currencies.
- Fit linear models and print coefficients and RMSE.

## 3. The NYC Taxi
- Use a shell command to determine how long it takes to **decompress** the data file and count how many lines it has.
- Use a shell command to count the number of lines in each compressed file for year 2020 in **parallel**.
- Use R to predict the number of records for each file in the directory. (**Sys.glob**, **file.info**)
- Plot the number of trips per month over time based on the predictions.
- Use a shell command to extract the field hvfhs_license_num and calculate the distribution of the number of trips per vendor for the months 2020-03 and 2021-03.
- Use a linear model with the trip time in hours and trip length in miles as predictors to model the driver pay.
- Process large data
- Use the **biglm** and **iotools** packages to fit a linear model.
- Use chunk.reader(xzfile(..., "rb")) to **decompress** and read the above file chunkwise.

## 5. Basic classification methods, data resampling techniques and regression trees
- Compute the **confusion matrix** and **misclassification rate**, using **linear discriminant analysis** and the **Naive Bayes method**.
- Compute the training and test misclassification rates.
- Use 10-fold cross-validation, with 20 repetitions, to compute the CV misclassification rates of **KNN**.
- Show in one graph the curve for these misclassification rates.
- Use the **Jackknifing** technique to find an appropriate value of K for KNN.
- Perform the Jackknifing selection of K using **parallel computing**, with the function **mclapply**().
- Fit an unpruned regression tree with rpart.
- Find the variation (**TSS**) reduction by the split at the root node.
- Find the predicted response value by the tree for the following observation.
- Find the **pruned tree**, using **1 SE-rule**.
- Find the optimal value of the complexity parameter.

## 6. Classification Trees and Ensemble Methods
- Grow an unpruned tree and prune it using the cost-complexity criterion.
- Pruning the tree using 10-fold cross-validation with 20 repetitions (in aid of **parallel computing** using 20 cores) for minimising the deviance (the **Gini index** by default).
- Produce a **Bagging** model for the training data with 500 trees constructed.
- Compute both the training and test errors of the Bagging predictor.
- Produce a **Random Forest** model with 500 trees constructed.
- Compute both the training and test errors of this Random Forest predictor.
- Produce a **Boosting** model, using up to 500 trees, with the number of trees determined via 10-fold cross-validation.
- Compute both the training and test errors of this Boosting predictor.

## 7. Clustering and Support Vector Machines
- In aid of **Random Forest**, find the two most important predictors out of 256 ones (in terms of Accuracy), for classifying between observations with digit = 6 and digit = 8.
- Select the first most important predictor.
- run the **K-means** algorithm to partition the images into clusters.
- Using four **linkage** methods: "complete", "single", "average" and "centroid".
- Produce a scatter plot for each of the following 6 partitioning results with 2 classes/clusters (class labels, K-means, complete linkage, single linkage, average linkage, centroid linkage).
- Train **support vector machines** using the linear kernel.
- Compute the training and **misclassification rates**.
- Use **radial** kernels for support vector machines. With cost=1 held fixed, train support vector machines, for gamma = 0.001, 0.01, 0.1, 1, 10, 100, respectively.
- Produce a classification plot for each value of gamma, which also shows the **decision boundary**.

## 8. Neural Networks and Deep Learning
- Train a **neural network** by minimising the cross-entropy objective function. The network has one **hidden layer** with 3 **units**.
- Train (without validation) a neural network with two hidden layers, with 2 and 3 units, respectively, by minimising the **cross-entropy** objective function.
- plot their **decision boundaries** inside a scatter plot of the training data with **jittering**.
- Use a **convolutional neural network** for predicting digit=6 or 8.
- Design and train a proper CNN and compute its training and test errors.
- Add two callbacks, one for **early stopping** (on validation accuracy) and the other for saving the best models only.
