import ast
import numpy as np
import pandas as pd
import sys

from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor

X_COLUMNS = ['length', 'age', 'is_favorite_genre', 'beach_read', 'biography',
                'classic', 'drama', 'history', 'pop_psychology', 'pop_sci',
                'romance', 'sci_fi', 'self_help', 'thriller']
Y_COLUMN = ['purchased']
AGES = ['0-17', '18-25', '26-35', '36-45', '46-55', '56-65', '66+', np.nan]

def combine_data(shipment, customers):
    """Combine customer and shipment data.

    shipment: dataframe of the month's shipment
    customers: dataframe of customer records
    """
    df = pd.merge(shipment, products, on='product_id')
    df = pd.merge(df, books, on='product_id')
    df = pd.merge(df, customers, on='customer_id')
    df['is_favorite_genre'] = favorite_genre(df['favorite_genres'], df['genre'])
    return df

def remove_data_errors(df):
    """Removes rows with invalid entries.

    df: dataframe containing book difficulty and length
    """
    df = df.loc[(df['difficulty'] >= 1) & (df['length'] < 1000)].reset_index()
    return df

def favorite_genre(favorites, genre):
    """Returns a 1 indicating that the genre of the book is a customer's
    favorite genre. Otherwise a 0 is returned. The inputs must be the same
    length.

    favorites: a list of lists containing a customers favorite genres
    genre: a list containing the genre of a shipped book
    """
    return [1 if x in favorites[ix] else 0 for ix, x in enumerate(genre)]

def create_model(df, neighbors):
    """Generates a nearest neighbors model.

    df: dataframe of shipped books
    neighbors: integer of neighbors to use in the model
    """
    X_train, X_test, y_train, y_test = train_test_split(df[Y_COLUMN],
                                                        df[X_COLUMNS],
                                                        test_size=.25)
    knn = KNeighborsRegressor(n_neighbors=100)
    knn.fit(y_train, X_train)
    return knn, y_test, X_test

def prediction(model, y_test, probability_cutoff):
    """Returns predictions about if a customer will buy the shipped book.

    model: nearest neighbor model
    y_test: dataframe of shipment and customer details
    probability_cutoff: decimal between 0 and 1
    """
    predictions = model.predict(y_test)
    return [True if x >= probability_cutoff else False for x in predictions]

def model_evaluation(actuals, predictions):
    """Returns the precision and recall of the model

    actuals: list of actual purchases
    predictions: list of predicted purchases
    """
    return classification_report(actuals, predictions)

def cash_flow(inventory, shipment, purchase_col):
    """Takes income from customer book purchases and subtracts expenses due to
    shipping and company book purchases.

    inventory: dataframe containing month's inventory
    shipment: dataframe containing month's shipment
    purchase_col: string indicating column used to determine purchases
    """

    shipped = shipment.groupby('product_id')
    shipped = shipped.aggregate({purchase_col: 'sum', 'customer_id': 'count'})
    shipped = shipped.reset_index()
    shipped = pd.merge(shipped, inventory[['product_id', 'retail_value']],
                       on='product_id')
    shipping_cost = ((shipped['customer_id']*2-shipped[purchase_col])*.6).sum()
    book_cost = (inventory['quantity_purchased']*inventory['cost_to_buy']).sum()
    expense = shipping_cost + book_cost
    income = (shipped[purchase_col]*shipped['retail_value']).sum()
    return income-expense

# load data sources
customers = pd.read_csv(sys.argv[1])#("customer_features.csv")
products = pd.read_csv(sys.argv[2])#"product_features.csv")
last_month_shipment = pd.read_csv(sys.argv[3])#("last_month_assortment.csv")
last_month_purchase_order = pd.read_csv(sys.argv[4])#"original_purchase_order.csv")
next_month_shipment = pd.read_csv(sys.argv[5])#"next_month_assortment.csv")
next_month_purchase_order = pd.read_csv(sys.argv[6])#("next_purchase_order.csv")

# clean up initial datasets
customers['favorite_genres'] = customers['favorite_genres'].apply(lambda x: x.lower().replace('-', '_'))
customers['favorite_genres'] = customers['favorite_genres'].apply(lambda x: ast.literal_eval(x))
customers['age'] = customers['age_bucket'].apply(lambda x: AGES.index(x))
products['genre'] = products['genre'].apply(lambda x: x.lower().replace('-', '_'))

# cast customer genre list to wide data
genres = []
for x in customers['favorite_genres']:
    for genre in x:
        genres.append(genre)
genres = np.unique(genres)

for genre in genres:
    new_col = []
    for favorite in customers['favorite_genres']:
        new_col.append(1 if genre in favorite else 0)
    customers[genre] = new_col

# create book price list
books = last_month_purchase_order.append(next_month_purchase_order)
books = books[['product_id', 'cost_to_buy', 'retail_value']].drop_duplicates()

# prepare purchase data for model
last_month = combine_data(last_month_shipment, customers)
last_month = remove_data_errors(last_month)

# create nearest neighbor model
model, y_test, X_test = create_model(last_month, 20)

#              precision    recall  f1-score   support
#
#       False       0.79      0.84      0.82      5188
#        True       0.65      0.58      0.62      2698
#
# avg / total       0.75      0.75      0.75      7886

# apply model to next month's shipment
next_month = combine_data(next_month_shipment, customers)
next_month['prediction'] = prediction(model, next_month[X_COLUMNS], .5)

# evaluate observed and projected cash flow
last_month_cash_flow = cash_flow(last_month_purchase_order, last_month_shipment,
                                 'purchased')
next_month_cash_flow = cash_flow(next_month_purchase_order, next_month,
                                 'prediction')

# determine if loan can be paid and new books ordered
if last_month_cash_flow+next_month_cash_flow >= 0:
    print('Yes')
else:
    print('No')
