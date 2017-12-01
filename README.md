# kaggle: Recruit Restaurant Visitor Forecasting
Use reservation and visitation data to predict the total number of visitors to a restaurant for future dates. 
Data and description can be found [here](https://www.kaggle.com/c/recruit-restaurant-visitor-forecasting).

**weighted_mean.py** predicts visitors using median for a restaurant in a given day of the week, and using weighted average (with higher weight for closer dates). Then takes arithmetric mean, geometric mean and harmonic mean of the two predictions. 
