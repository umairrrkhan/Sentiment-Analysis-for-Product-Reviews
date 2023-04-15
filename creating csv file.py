import pandas as pd

# Create a dictionary with review data
data = {
    'review_text': [
        "I loved this product, it works really well!",
        "This was a terrible purchase, I wouldn't recommend it to anyone.",
        "I'm on the fence about this product, it has some pros and cons.",
        "This product exceeded my expectations, I highly recommend it."
    ],
    'sentiment': ['positive', 'negative', 'neutral', 'positive']
}

# Create a pandas dataframe from the dictionary
df = pd.DataFrame(data)

# Save the dataframe to a CSV file
df.to_csv('reviews.csv', index=False)