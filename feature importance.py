##### Feature importance with order 
##### Take NB model as an example 

from sklearn.inspection import permutation_importance
model = MultinomialNB()

# fit the model
model.fit(X_train, y_train)

# perform permutation importance
results = permutation_importance(model, X_train, y_train, scoring='accuracy')

# get importance
importance = results.importances_mean

# summarize feature importance
for i,v in enumerate(importance):
    print('Feature: %0d, Score: %.5f' % (i,v))


name = X_train.columns
idx = (-importance).argsort()
desc_feature = [name[i] for i in idx]

top_feature = desc_feature[:10]
print(top_feature)

score = list(importance)
score.sort(reverse = True)

# ready plot
fig, ax = plt.subplots(figsize = (16,9))
ax.barh(top_feature, score[0:10])
ax.tick_params(axis = 'x', labelsize =18)
ax.tick_params(axis = 'y', labelsize =26)
