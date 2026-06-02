# Facial-Recognition
Written in Python, the goal of this project was to build a supervised learning model using PCA and K-means clustering
in order to explore and analyze features from contestants of the popular show "Survivor", in comparison with our school's Computer Science faculty.
This model learns key features in the contestant's faces to learn what a winner most closely looks like, and which professor looks most likely to win in the next season of Survivor.
Furthermore, we explore with of the professors looks LEAST like a face by reconstructing their faces through PCA and measuring their Euclidean distances to their original faces.
To determine which professor is most likely to be the next HOST of Survivor, we projected each professor into a "face space", then applied nearest neighbor classification in comparison with the host, Jeff Probst.
We investigated which season each professor would most likely be on by using K-means on the previously PCA reduced survivor faces and assigning each PCA professor to their closest cluster.
Lastly, to determine who would have the greatest odds of winning the next season of Survivor,
we took the mean average weights of all previous winners of Survivor. Then, each professor is compared to the winner's mu via the euclidean distance formula.
Whichever professor has the shortest distance (smallest value) is predicted to be the next winner
