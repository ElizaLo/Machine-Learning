# K-Means Clustering

## Complexity

The average complexity is given by _O(k n T)_, where _n_ is the number of samples and _T_ is the number of iteration.

The worst case complexity is given by _O(n^(k+2/p))_ with _n = n_samples_, _p = n_features_. (D. Arthur and S. Vassilvitskii, ‘How slow is the k-means method?’ SoCG2006)

## Advantages

1. In practice, the k-means algorithm is very fast (one of the fastest clustering algorithms available).

## Disadvantages

1. Complexity
2. It falls in local minima. That’s why it can be useful to restart it several times.

## Articles 

- [K-Means Clustering in Python: A Practical Guide](https://realpython.com/k-means-clustering-python/) by Real Python
