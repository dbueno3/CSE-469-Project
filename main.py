import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

''' 
Classification: Predict if a Pokémon is Legendary based on its stats. 
or
Clustering: Group Pokémon into clusters based on similar attributes.
'''

def load_data(file_path):
    return pd.read_csv(file_path)

def preprocess_data(df, categorical_features):
    
    # Define the one-hot encoder for the categorical features
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', categorical_transformer, categorical_features)
        ])

    return preprocessor

def train_model(X_train, y_train, preprocessor):

    model = Pipeline(steps=[('preprocessor', preprocessor),
                            ('classifier', RandomForestClassifier(random_state=42))])
    # Train the model
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    # Predictions
    y_pred = model.predict(X_test)
    # Evaluation
    print(classification_report(y_test, y_pred))
    return y_pred

def plot_feature_importances(model, categorical_features):
    # Feature Importances
    ohe = model.named_steps['preprocessor'].named_transformers_['cat'].named_steps['onehot']
    feature_names = ohe.get_feature_names_out(input_features=categorical_features)
    feature_importances = model.named_steps['classifier'].feature_importances_
    indices = np.argsort(feature_importances)[::-1]

    # Plotting
    plt.figure(figsize=(12, 6))
    plt.title('Feature Importances')
    plt.bar(feature_names[indices], feature_importances[indices], align='center')
    plt.xticks(rotation=90)
    plt.show()

def plot_average_stats_by_legendary_dot(df):
    # Calculate the mean of the stats for each group (Legendary and non-Legendary)
    average_stats = df.groupby('Legendary')[['HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed']].mean().T

    # Plotting
    plt.figure(figsize=(10, 6))

    # Plot each stat as a separate line
    for column in average_stats.columns:
        plt.plot(average_stats.index, average_stats[column], 'o-', label=f"Legendary: {column}")

    plt.title('Average Stats by Legendary Status')
    plt.xlabel('Stats')
    plt.ylabel('Average Value')
    plt.legend(title='Status')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_average_stats_by_legendary(df):
    # Calculate the mean of the stats for each group (Legendary and non-Legendary)
    average_stats = df.groupby('Legendary')[['HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed']].mean()

    # Plot the averages for each stat by Legendary status
    average_stats.plot(kind='bar', color=['grey', 'darkorange'], alpha=0.7)

    plt.title('Average Stats by Legendary Status')
    plt.xlabel('Legendary Status')
    plt.ylabel('Average Stat Value')
    plt.xticks(ticks=[0, 1], labels=['Non-Legendary', 'Legendary'], rotation=0)
    plt.legend(title='Stats')
    plt.grid(axis='y')
    plt.tight_layout()
    plt.show()

def plot_avg_stats_comparison(df):
    # Calculate the average of the selected stats for each Pokémon
    df['Avg_Physical'] = df[['Attack', 'Sp. Atk', 'Speed']].mean(axis=1)
    df['Avg_Special'] = df[['Defense', 'Sp. Def', 'HP']].mean(axis=1)

    # Separate the data into Legendary and non-Legendary Pokémon
    legendary = df[df['Legendary'] == True]
    non_legendary = df[df['Legendary'] == False]

    # Create the scatter plot
    plt.figure(figsize=(10, 6))
    plt.scatter(non_legendary['Avg_Physical'], non_legendary['Avg_Special'], label='Non-Legendary', alpha=0.7)
    plt.scatter(legendary['Avg_Physical'], legendary['Avg_Special'], color='darkorange', label='Legendary', alpha=0.7)

    plt.title('Average Attack vs. Defensive Stats')
    plt.xlabel('Attack/Sp. Atk/Speed')
            #    Speed/Average HP')
    plt.ylabel('Average Defense/Sp. Def/HP')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_confusion_matrix(y_test, y_pred):
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

file_path = './pokemon.csv'
categorical_features = ['Type 1', 'Type 2']  

# Load the dataset
pokemon_df = load_data(file_path)
preprocessor = preprocess_data(pokemon_df, categorical_features)
X = pokemon_df.drop('Legendary', axis=1)
y = pokemon_df['Legendary']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = train_model(X_train, y_train, preprocessor)

# Evaluate the model
y_pred = evaluate_model(model, X_test, y_test)

# # Plot the feature importances
# plot_feature_importances(model, categorical_features)

# # Assume pokemon_df is your DataFrame after loading the data
# plot_pokemon_dual_types_legendary_colored(pokemon_df)

# # Call the function with the DataFrame
# plot_average_stats_by_legendary_dot(pokemon_df)


plot_avg_stats_comparison(pokemon_df)

# # Plot the confusion matrix
# plot_confusion_matrix(y_test, y_pred)