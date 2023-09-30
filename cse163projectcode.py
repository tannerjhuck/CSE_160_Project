"""
Jaiden Atterbury and Tanner Huck
CSE 163 AD
05/25/23

CSE 163 Final Project Code:
Implements the functions and runs the code necessary to answer our research
questions.
"""

# Import necessary packages and functions/classes:
import geopandas as gpd
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import statsmodels.api as sm
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import linear_rainbow
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.api import het_breuschpagan
from statsmodels.tools.eval_measures import mse

# Set seed for reproducibility:
np.random.seed(10)


def explotatory_line_avg_life(data: pd.DataFrame) -> None:
    """
    Takes in the life expectany data set and graphs a line plot of
    the average life expectancy over time. In particular over time
    means for every year present in the dataset. The function saves
    the plot to a file called average_life_expectancy_over_time.png.
    """
    # Finding average life expectancy per year:
    life_expectancy_by_year = data.groupby('Year')['Life_expectancy'].mean()

    # Creating a line plot with each pont as a dot:
    plt.figure(figsize=(10, 6))
    plt.plot(life_expectancy_by_year.index,
             life_expectancy_by_year.values, marker='o')

    # Add axis labels and a title:
    plt.xlabel('Year')
    plt.ylabel('Average Life Expectancy')
    plt.title('Average Life Expectancy Over Time')

    # Save the plot as a png:
    plt.savefig('average_life_expectancy_over_time.png', bbox_inches='tight')


def exploratory_bar_best_worst_country(data: pd.DataFrame) -> None:
    """
    Takes in the life expectany data set and grpahs a bar plot of
    the five best and five worst countries by average life expectancy.
    The countries with the highest life expectancy will be plotted
    higher along the y-axis. The function save the resulting plot to
    a file called best_worst_countries.png.
    """
    # Collect the life expectancy for each country in a sorted DataFrame:
    life_by_country = data.groupby('Country')['Life_expectancy']
    life_by_country = life_by_country.mean().sort_values()

    # Find the 5 countries with the highest and lowest life expectancy
    # in decensding order (for purpose of graphing):
    top_5 = life_by_country.tail(5).sort_values(ascending=False)
    bottom_5 = life_by_country.head(5).sort_values(ascending=False)

    # Combine the top and bottom five countries together:
    combined_df = pd.concat([top_5, bottom_5])

    # Creating a bar plot with the top 5 countires higher on the y-axis
    # and bottom 5 lower on the y-axis:
    plt.figure(figsize=(10, 6))
    plt.barh(combined_df.index[::-1], combined_df.values[::-1])

    # Add axis labels and a title:
    plt.xlabel('Average Life Expectancy')
    plt.title('Top 5 Highest and Lowest Average Life Expectancy by Country')

    # Save the plot as a png:
    plt.savefig('best_worst_countries.png', bbox_inches='tight')


def exploratory_scatter_diff_factors(data: pd.DataFrame) -> None:
    """
    Takes in the life expectany data set and plots 5 different scatter
    plots all in one figure. Each scatter plot will display the
    relationship between a certain varable and life expectancy.
    The variables that we will look at are 'Infant Deaths',
    'Under Five Deaths', 'Adult Mortality', 'BMI', 'GDP_per_capita'.
    There will also be a line of best fit on each scatter plot. The
    function will save the resulting plot to the file named
    facet_factors.png.
    """
    # Create the list of variables that we want to graph
    factors = ['Life_expectancy', 'Infant Deaths', 'Under Five Deaths',
               'Adult Mortality', 'BMI', 'GDP_per_capita']

    # Get the DataFrame that only has the columns listed in factos:
    subset_data = data[factors]

    # Drop missing values from  the subset:
    subset_data = subset_data.dropna()

    # Creating the plotting figure:
    plt.figure(figsize=(15, 4))

    # Add the title for overall figure:
    plt.suptitle('Factors affecting Life Expectancy', fontweight='bold')

    # Loop through each factor that we want to graph:
    for i, factor in enumerate(factors[1:], start=1):
        # Create a subplot in the figure:
        plt.subplot(1, 5, i)

        # Ploting the variable vs life expectancy in a scatter plot (blue dots)
        # with a line of best fit (red):
        sns.regplot(data=subset_data, x=factor, y='Life_expectancy',
                    scatter_kws={'color': 'blue'}, line_kws={'color': 'red'})
        if i == 1:
            # If this is the first subplot, add a y-axis label:
            plt.ylabel('Life Expectancy', fontweight='bold')
        else:
            # Otherwise don't add a y-axis label:
            plt.gca().set_ylabel('')

        # Add an x-axis label and y-axis domain to each variable:
        plt.xlabel(factor if factor != 'GDP_per_capita' else 'GDP per capita',
                   fontweight='bold')
        plt.ylim(30, 90)

    # Save the plot to a png
    plt.savefig('facet_factors.png')
    plt.tight_layout()


def corr_matrix(cont_columns: pd.DataFrame) -> pd.DataFrame:
    """
    Takes in a DataFrame called cont_columns representing the continuous
    columns of the life expectancy data set and returns a DataFrame
    representing the correlation matrix between all of the continuous variables
    in the data set.
    """
    # Create and return the correlation matrix:
    corr_matrix = cont_columns.corr()

    return corr_matrix


def get_density(data: pd.Series, img_name: str, axis: str,
                title: str) -> None:
    """
    Takes in a Series called data representing either the life expectancy data
    itself or residuals from a linear model based on the life expectancy data
    as well as a String called img_name, axis, and title. The function creates
    and saves a displot to img_name.png with x-axis label using the axis
    parameter and with a title using the parameter title.
    """
    # Get the image string:
    img_str = img_name + '.png'

    # Plot the data to a displot using seaborn:
    sns.displot(data, kde=True)

    # Add axis and title labels:
    plt.xlabel(axis)
    plt.title(title)

    # Save the image to a png:
    plt.savefig(img_str)


def get_mlr(y: pd.Series, X: pd.DataFrame, constant: bool = True):
    """
    Takes in a Series y representing the dependent variable of the model and
    a DataFrame X representing the independent variables of the model, as well
    as a bool called constant which defualts as True representing if a
    constant parameter should be added to the model. The function then fits
    and returns a multiple linear regression model to this data.
    """
    # Add constant if specified:
    if constant:
        X = sm.add_constant(X)

    # Fit and return the model:
    model = sm.OLS(y, X).fit()

    return model


def get_mlr_results(model, X_train: pd.DataFrame, Y_train: pd.Series,
                    X_test: pd.DataFrame, Y_test: pd.Series,
                    constant: bool = True) -> tuple[str, str]:
    """
    Takes in a statsmodel linear regression model named model, and a training
    and test data set, as well as a bool called constant which defualts as
    True representing if a constant parameter should be added to the model.
    The function returns a tuple of strings reporting the training and testing
    mean squared error value.
    """
    # Add constant if specified:
    if constant:
        X_train = sm.add_constant(X_train)
        X_test = sm.add_constant(X_test)

    # Get training predictions:
    train_predict = model.predict(X_train)

    # Get training mean squared error:
    mse_train = mse(train_predict, Y_train)

    # Get testing predictions:
    test_predict = model.predict(X_test)

    # Get testing mean squared error:
    mse_test = mse(test_predict, Y_test)

    # Return the tuple reporting the error:
    return ("Train mean squared error: {}".format(mse_train),
            "Test mean squared error: {}".format(mse_test))


def get_mlr_tests(model) -> tuple[str, str, str]:
    """
    Takes in a statsmodel linear regression model named model and returns
    a tuple of strings cotaining the results from a rainbow test, a
    durbin-watson test, and a breuschpagen test. These tests are used to test
    the regression assumptions
    """
    # Rainbow test for linearity:
    rainbow_test = linear_rainbow(model)[1]

    # Durbin-Watson test for autocorrelation:
    dw_test = durbin_watson(model.resid)

    # Breuschpagan test for homoscedasticity:
    hbp_test = het_breuschpagan(model.resid, model.model.exog)[1]

    # Return the tuple containg the test information:
    return ("Rainbow test p-value: {}".format(rainbow_test),
            "Durbin-Watson test statistic: {}".format(dw_test),
            "Breuschpagan test p-value: {}".format(hbp_test))


def get_log(y: pd.Series, X: pd.DataFrame, constant: bool = True):
    """
    Takes in a Series y representing the dependent variable of the model and
    a DataFrame X representing the independent variables of the model, as well
    as a bool called constant which defualts as True representing if a
    constant parameter should be added to the model. The function then fits
    and returns a logistic regression model to this data.
    """
    # Add constant if specified:
    if constant:
        X = sm.add_constant(X)

    # Fit and return the model:
    logit_mod = sm.Logit(y, X).fit()

    return logit_mod


def log_results_helper(model, y: pd.Series, X: pd.DataFrame) -> float:
    """
    Helper function for the get_log_results function. It takes in a logistic
    regression model called model, a dependent variable called y, and
    independent variables called X and returns a float representing the
    accuracy score for that model.
    """
    # Get the predicted y values:
    Y_hat = list(model.predict(X))

    # Turn these predicions to 0 or 1 for classification:
    prediction = list(map(round, Y_hat))

    # Compute and return the accuracy score of the model
    accuracy = accuracy_score(prediction, y)

    return accuracy


def get_log_results(model, X_train: pd.DataFrame, Y_train: pd.Series,
                    X_test: pd.DataFrame, Y_test: pd.Series,
                    constant: bool = True) -> tuple[str, str]:
    """
    Takes in a statsmodel logistic regression model named model, and a training
    and test data set, as well as a bool called constant which defualts as
    True representing if a constant parameter should be added to the model.
    The function returns a tuple of strings reporting the training and testing
    model accuracy score.
    """
    # Add constant if specified:
    if constant:
        X_train = sm.add_constant(X_train)
        X_test = sm.add_constant(X_test)

    # Get training set accuracy score:
    train_accuracy = log_results_helper(model, Y_train, X_train)

    # Get testing set accuracy score:
    test_accuracy = log_results_helper(model, Y_test, X_test)

    # Return the tuple reporting the accuracy scores:
    return ("Train accuray: {}".format(train_accuracy),
            "Test accuracy: {}".format(test_accuracy))


def get_decision_results(X_train: pd.DataFrame, Y_train: pd.Series,
                         X_test: pd.DataFrame,
                         Y_test: pd.Series) -> tuple[str, str]:
    """
    Takes in a traiing and test set and creates a decision tree classifier
    and returns the training and testing accuracy of this decision tree
    classfier as a tuple of strings.
    """
    # Create an untrained model:
    model = DecisionTreeClassifier()

    # Train it on the training set:
    model.fit(X_train, Y_train)

    # Compute training accuracy:
    train_predictions = model.predict(X_train)
    train_acc = accuracy_score(Y_train, train_predictions)

    # Compute test accuracy:
    test_predictions = model.predict(X_test)
    test_acc = accuracy_score(Y_test, test_predictions)

    # Return the tuple reporting the accuracy scores:
    return ("Train accuray: {}".format(train_acc),
            "Test accuracy: {}".format(test_acc))


def get_vif(vars: pd.DataFrame) -> pd.DataFrame:
    """
    Takes in a DataFrame called vars representing some of the variables from
    the life expectancy data set and calculates and returns a DataFrame of the
    variance inflation factor of each of the variables in the given DataFrame.
    """
    # Create the template for the VIF DataFrame:
    vif_data = pd.DataFrame()

    # Add the variables names into the column named Feature:
    vif_data["Feature"] = vars.columns

    # Calculating VIF for each feature and adding it to the DataFrame:
    vif_data["VIF"] = [variance_inflation_factor(vars.values, i)
                       for i in range(len(vars.columns))]

    return vif_data


def stundentized_resid(model) -> None:
    """
    Takes in a linear regression model from the statsmodels package and plots
    and saves a studentized residual plot to resid_plot.png. In particular
    this plot is meant to be the Python analgous to ols_plot_resid_stud in the
    R programming language.
    """
    # Get the studentized residuals from the model:
    stud_res = model.outlier_test()

    # Isolate the studentized residual column:
    resid = stud_res['student_resid']

    # Create subplots and axes:
    fig, ax = plt.subplots(1)

    # Loop though the studentized residuals:
    for index, value in enumerate(resid.values):
        # If the absolute value of the residual is greater than or equal to
        # three plot a red vertical line equal in magnitutde to that residual:
        if abs(value) >= 3:
            ax.plot(np.array([index, index]), np.array([0, value]), color='r',
                    linewidth=0.5, label='outlier')
            # Provide the accompanying index of the outlier value, if-else is
            # needed for text formatting purposes:
            if value >= 0:
                ax.text(index, value-0.01, str(index), ha='left', va='bottom',
                        fontsize=5, color='k')
            else:
                ax.text(index, value-0.01, str(index), ha='left', va='center',
                        fontsize=5, color='k')
            # If not plot a blue vertical line equal in magnitude to that
            # residual:
        else:
            ax.plot(np.array([index, index]), np.array([0, value]), color='b',
                    linewidth=0.5, label='normal')

    # Plot a horizontal line at y= -3, -2, -1, 0, 1, 2, 3. At y = -3, 3, make
    # the line red. Otherwise the line should be black:
    for val in range(4):
        if val == 0:
            ax.axhline(y=val, color='k', linewidth=0.5)
        elif val in [1, 2]:
            ax.axhline(y=val, color='k', linewidth=0.5)
            ax.axhline(y=-val, color='k', linewidth=0.5)
        else:
            ax.axhline(y=val, color='r', linewidth=0.5)
            ax.axhline(y=-val, color='r', linewidth=0.5)

    # Add descriptive threshold text to the top right corner:
    ax.text(0.99, 0.99, "Threshold: abs(3)", ha='right', va='top',
            transform=ax.transAxes, fontsize=7, color='m')

    # Add x-axis and y-axis labels as well as a title:
    ax.set_title("Studentized Residuals Plot", loc='left')
    ax.set_ylabel("Deleted Studentized Residuals")
    ax.set_xlabel("Observation")

    # Create custom legend handles and labels:
    legend_handles = [
        plt.Line2D([], [], color='blue', marker='s', markersize=5,
                   linestyle='None'),
        plt.Line2D([], [], color='red', marker='s', markersize=5,
                   linestyle='None')
    ]

    legend_labels = ['Normal', 'Outlier']

    # Create the legend:
    ax.legend(legend_handles, legend_labels, title="Observation",
              bbox_to_anchor=(1.0001, 0.5), loc='center left')

    plt.savefig("stud_resid.png")


def interactive_lineplot_continent(data: pd.DataFrame) -> None:
    """
    Takes in the merged countries and life expectany data set.
    Creates an interactive lineplot that shows each continent
    as a different line that measures life expectancy over time.
    Interaction will allow user to hover over each line and view
    current continent, life expectancy, and year. The function
    write the plot to an html file called interactive_line_graph.html.
    """
    # Collect the life expectacny for each continent per year:
    life_by_continent = data.groupby(['CONTINENT', 'Year'])['Life_expectancy']
    life_by_continent = life_by_continent.mean().reset_index()

    # Create a interactive line plot of life expectancy vs year coloring each
    # line by continent:
    fig = px.line(life_by_continent, x='Year', y='Life_expectancy',
                  color='CONTINENT',
                  title='Average Life Expectancy by Continent Over Time',
                  labels={'Year': 'Year',
                          'Life_expectancy': 'Life Expectancy'})

    # Add a legend to the right-middle of the plot:
    fig.update_layout(legend=dict(orientation='h', yanchor='bottom', y=1.02,
                                  xanchor='right', x=1))

    # Save the plot to an  html file:
    pio.write_html(fig, file='interactive_line_graph.html')


def interactive_bar_region(data: pd.DataFrame) -> None:
    """
    Takes in the merged countries and life expectany data set.
    Creates an interactive bar plot with each bar as a differnt region
    of the world that measures their average life expectancy.
    Interaction allows user to hover over each bar and see which region
    a bar is and what their average life expectancy is. The function
    save the plot to a file called 'interactive_bar_plot.html'.
    """
    # Find the averge life expectancy in each region of the world:
    life_expectancy_region = data.groupby('Region')['Life_expectancy']
    life_expectancy_region = life_expectancy_region.mean().reset_index()

    # Create an interactive bar plot of average life expectancy vs region:
    fig = go.Figure(data=[go.Bar(
        x=life_expectancy_region['Region'],
        y=life_expectancy_region['Life_expectancy'],

        # Add interaction that will allow user to hover over bars and will
        # display which region it is and the avgerage life expectancy rounded
        # to 2 decimals:
        hovertext=life_expectancy_region['Region'] +
        '<br>Average Life Expectancy: ' +
        life_expectancy_region['Life_expectancy'].round(2).astype(str),

        # Adds s blue color scale to show which regions have relativily higher
        # avgerage life expectancy:
        marker=dict(color=life_expectancy_region['Life_expectancy'],
                    colorscale='YlGnBu')
    )])

    # Add axis labels and a title:
    fig.update_layout(
        xaxis=dict(title='Region', title_font=dict(color='black')),
        yaxis=dict(title='Average Life Expectancy',
                   title_font=dict(color='black')),
        title=dict(text='Average Life Expectancy by Region', x=0.5,
                   font=dict(color='black'))
    )

    # Remove unwanted label in interaction details:
    fig.update_traces(
        hovertemplate='%{x}<br>Average Life Expectancy: %{y}<extra></extra>')

    # Save the plot to an html file:
    pio.write_html(fig, file='interactive_bar_plot.html')


def interactive_map_country(data: pd.DataFrame) -> None:
    """
    Takes in the merged countries and life expectany data set.
    Creates an interactive cloropleth map that shows each country
    on a map colored by their average life expectancy. Interaction
    allows user to hover over each country and see what their exact
    average life expectancy is. The function save the graph to a file
    named world_map.html.
    """
    # Create and  interactive choropleth map of the world:
    fig = px.choropleth(
        data,
        # Graphing each country:
        locations='NAME',
        locationmode='country names',

        # Color each country on a blue color scale according to avgerage life
        # expectancy
        color='Life_expectancy',
        color_continuous_scale='YlGnBu',

        # Use natural earth map style:
        projection='natural earth',

        # Details that what will be shown through interaction:
        labels={'Life_expectancy': 'Life Expectancy (years)',
                'NAME': 'Country'}
    )

    # Adding labels, title, and a legend to the graph:
    fig.update_layout(
        title=dict(
            text='Life Expectancy by Country',
            font=dict(size=28)
        ),
        coloraxis_colorbar=dict(
            len=0.5,
            yanchor='middle',
            y=0.5
        )
    )

    # Save to an html file:
    pio.write_html(fig, file='world_map.html')


def main():
    # Read in the life expectancy data:
    life_expect = pd.read_csv("LifeExp.csv")

    # Read in the shape file and create mergedd data with the shape file
    # and life expectancy data set
    world_shapefile = gpd.read_file('geo_data/ne_110m_admin_0_countries.shp')
    merged_data = world_shapefile.merge(life_expect, left_on='ADMIN',
                                        right_on='Country')

    # Summary statistics
    life_expectancy_summary = life_expect['Life_expectancy'].describe()
    print(life_expectancy_summary)

    # Data exploration:
    explotatory_line_avg_life(life_expect)
    exploratory_bar_best_worst_country(life_expect)
    exploratory_scatter_diff_factors(life_expect)

    # Extract the continuous/non-categorial columns from the DataFrame:
    cont_columns = life_expect.iloc[:, [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14,
                                        15, 16, 17, 18, 21]]

    # Create and save the correlation matrix of the dataset:
    correlation = corr_matrix(cont_columns)
    correlation.to_csv('corr_matrix.csv')

    # Get the density plot of life expectancy:
    life_data = life_expect['Life_expectancy']
    get_density(life_data, "life_dist", "Life Expectancy",
                "Distribution of Life Expectancy")

    # Reserach question #1 : Multiple Linear Regression:

    # Remove life expectancy from the continuous columns DataFrame:
    cont_columns = cont_columns.iloc[:, :-1]

    # Split this data into a training and test set:
    X_train, X_test, Y_train, Y_test = train_test_split(cont_columns,
                                                        life_data,
                                                        test_size=0.2)

    # Find which variables are significant predictors of life expectancy:
    mlr_model_1 = get_mlr(Y_train, X_train)

    # Get the model summary:
    print(mlr_model_1.summary())

    # Significant predictors before VIF:
    sig_pred_before = cont_columns.iloc[:, [0, 1, 2, 3, 4, 6, 9, 10, 12, 14]]

    # Get the variance inflation factor of the significant predictors:
    sig_vif_before = get_vif(sig_pred_before)
    print(sig_vif_before)

    # Significant predictors after VIF:
    sig_pred_after = sig_pred_before.iloc[:, [2, 3, 6, 7, 8]]

    # Get the variance inflation factor of the new significant predictors:
    sig_vif_after = get_vif(sig_pred_after)
    print(sig_vif_after)

    # Split this data into a training and test set:
    X_train, X_test, Y_train, Y_test = train_test_split(sig_pred_after,
                                                        life_data,
                                                        test_size=0.2)

    # Create the final model:
    mlr_model_2 = get_mlr(Y_train, X_train)

    # Get the model summary:
    print(mlr_model_2.summary())

    # Test the model assumptions:

    # Test reults:
    assumption_results = get_mlr_tests(mlr_model_2)
    print(assumption_results)

    # Normality of residuals:
    get_density(mlr_model_2.resid, "resid_dist", "Residuals",
                "Distribution of Regression Residuals")

    # Outliers:
    stundentized_resid(mlr_model_2)

    # Model accuracy:
    mlr_accuracy = get_mlr_results(mlr_model_2, X_train, Y_train, X_test,
                                   Y_test)
    print(mlr_accuracy)

    # Research question #2: Logistic Regression:

    # Get the data for classification:
    y = life_expect['Economy_status_Developed']

    # Split this data into a training and test set:
    X_train, X_test, Y_train, Y_test = train_test_split(cont_columns, y,
                                                        test_size=0.2)

    # Logistic regression:

    # Find which variables are significant predictors of devloped/developing
    # label:
    logit_mod_1 = get_log(Y_train, X_train)

    # Get the model summary:
    print(logit_mod_1.summary())

    # Get significant predictors for the logistic model:
    X = life_expect.iloc[:, [4, 5, 6, 7, 9, 13, 14, 18]]

    # Split this data into a training and test set:
    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2)

    # Get the final model:
    logit_mod_2 = get_log(Y_train, X_train, constant=False)

    # Get the model summary:
    print(logit_mod_2.summary())

    # Model accuracy:
    log_accuracy = get_log_results(logit_mod_2, X_train, Y_train, X_test,
                                   Y_test, constant=False)
    print(log_accuracy)

    # Decision tree classifier:
    dt_model_results = get_decision_results(X_train, Y_train, X_test, Y_test)
    print(dt_model_results)

    # Reserach question #3: Life Expectancy Plotting:

    # Interacitve graphs
    interactive_lineplot_continent(merged_data)
    interactive_bar_region(merged_data)
    interactive_map_country(merged_data)

    plt.show()


if __name__ == '__main__':
    main()
