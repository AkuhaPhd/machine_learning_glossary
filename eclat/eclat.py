import pandas as pd
from apyori import apriori

# Load dataset
df = pd.read_csv("./data/Market_Basket_Optimisation.csv", header=None)

# Data transformed to list of transactions as Apriori model takes only a list
transactions = [map(str, i) for i in df.values]

# Train the Eclat model on the transaction dataset
rules = apriori(
    transactions=transactions,
    min_support=0.003,
    min_confidence=0.2,
    min_lift=3,
    min_length=2,
    max_length=2,
)

# Visualising the results
results = list(rules)
print(results)


# Putting the results well organized into a Pandas DataFrame
def inspect(results):
    lhs = [tuple(result[2][0][0])[0] for result in results]
    rhs = [tuple(result[2][0][1])[0] for result in results]
    supports = [result[1] for result in results]
    return list(zip(lhs, rhs, supports))


results_in_data_frame = pd.DataFrame(inspect(results),
                                     columns=['Product 1', 'Product 2', 'Support'])

# Displaying the results non-sorted
print(results_in_data_frame)

# Displaying the results sorted by descending lifts
print(results_in_data_frame.nlargest(n=10, columns='Support'))
