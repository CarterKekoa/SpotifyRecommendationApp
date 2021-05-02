"""
Programmer: Armando Valdez
Class: CptS 322-02, Spring 2021
Programming Assignment #3
2/25/21

Description: This program holds reusable utility functions
"""
import math
import copy
from mysklearn.mypytable import MyPyTable
def get_frequencies(table, col_name):
    """Get the total number of frequecies for every unique value in a column

        Args:
            table(MyPyTable): Table used to get the column from
            col_names(str): Name of the column we want to get from table

        Returns:
            values: A list of all unique values in the specific column
            counts: list of total occurances for each unique value in values.
    """
    col = MyPyTable.get_column(table, col_name, False)

    values = []
    counts = []

    for val in col:
        if val == "N/A":
            pass
        elif val not in values:
            values.append(val)
            counts.append(1)
        else:
            index = values.index(val)
            counts[index] += 1

    return values, counts

def get_averge_sales(table, col_name):
    """Gets the total average for a specific column of ints or floats

        Args:
            table(MyPyTable): Table used to get the column from
            col_names(str): Name of the column we want to get from table

        Returns:
            col_avg: an int/float that holds the column average after calculations
    """
    col_index = table.column_names.index(col_name)
    col_sum = 0
    num_col = 0
    for row in table.data:
        if row[col_index] != "NA" or row[col_index] != "N/A":
            num_col += 1
            col_sum += row[col_index]
    col_avg = (col_sum / num_col) * 100
    col_avg = round(col_avg, 2)
    return col_avg

def MPG_rating(table, col_name):
    """Sorts MPG values into there specific categories

        Args:
            table(MyPyTable): Table used to get the column from
            col_names(str): Name of the column we want to get from table

        Returns:
            count_list: A list containing multiple list of all the MPG values in there designated section
    """
    col = MyPyTable.get_column(table, col_name, False)
    categories = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    sub_tables = [[0] for _ in categories]
    count_list = []

    for val in col:
        if val <= 13:
            sub_tables[0][0] += 1
        elif val == 14:
            sub_tables[1][0] += 1
        elif val >= 15 and val <= 16:
            sub_tables[2][0] += 1
        elif val >= 17 and val <= 19:
            sub_tables[3][0] += 1
        elif val >= 20 and val <= 23:
            sub_tables[4][0] += 1
        elif val >= 24 and val <= 26:
            sub_tables[5][0] += 1
        elif val >= 27 and val <= 30:
            sub_tables[6][0] += 1
        elif val >= 31 and val <= 36:
            sub_tables[7][0] += 1
        elif val >= 37 and val <= 44:
            sub_tables[8][0] += 1
        elif val >= 45:
            sub_tables[9][0] += 1

    for index in sub_tables:
        count_list.append(index[0])
 
    return count_list

def compute_equal_width_cutoffs(values, num_bins):
    """Computes equal width bins for a modified bar chart that represents a histogram

        Args:
            values: List of all values that will be bined
            num_bins: The number of bins desired

        Returns:
            cutoffs: List of all bins cuttoffs 
    """
    values_range = max(values) - min(values)
    bin_width = values_range / num_bins 
    cutoffs = []
    cutoffs.append(min(values))
    for i in range(num_bins):
        cutoffs.append(cutoffs[i] + bin_width)
    cutoffs = [round(cutoff, 2) for cutoff in cutoffs]
    return cutoffs 

def compute_bin_frequencies(values, cutoffs):
    """Places each value into their designated bin

        Args:
            values: List of all values that will be bined
            cuttoffs: List of all the bin cuttofs 

        Returns:
            freqs: List of all the values seperated into each of their bins 
    """
    freqs = [0 for _ in range(len(cutoffs) - 1)]

    for val in values:
        if val == max(values):
            freqs[-1] += 1
        else:
            for i in range(len(cutoffs) - 1):
                if cutoffs[i] <= val < cutoffs[i + 1]:
                    freqs[i] += 1

    return freqs

def calc_regression(x, y):
    """Computes the regression line for a scatter plot

        Args:
            x:List of x-axis values
            y:List of y-axis values

        Returns:
            m: value needed to salve equation y = mx + b
            b: second value needed to solve equation y = mx + b
    """
    x_median = sum(x)/len(x)
    y_median = sum(y)/len(y)
    n = len(x)

    m_numer = sum([xi*yi for xi,yi in zip(x, y)]) - n * x_median * y_median
    m_denom = sum([xi**2 for xi in x]) - n * x_median**2

    m = m_numer / m_denom
    b = y_median - m * x_median

    return m, b


def calc_correlation_coeficient(x, y):
    """Computes the correlation coeficient of a scatter plot
        Args:
            x:List of x-axis values
            y:List of y-axis values

        Returns:
            corr_coe: Correlation coeficient value rounded to 2 decimal places
    """
    x_median = sum(x)/len(x)
    y_median = sum(y)/len(y)
    n = len(x)

    r_numer = sum([xi*yi for xi,yi in zip(x, y)]) - n * x_median * y_median
    r_denom = math.sqrt((sum([xi**2 for xi in x]) - n * x_median**2)*(sum([yi**2 for yi in y]) - n * y_median**2))

    corr_coe = r_numer / r_denom
    
    return round(corr_coe, 2)

def calc_covariance(x, y):
    """Computes the covariance of a scatter plot

        Args:
            x:List of x-axis values
            y:List of y-axis values

        Returns:
            cov: Covariance rounded to 2 decimal places
    """
    x_median = sum(x)/len(x)
    y_median = sum(y)/len(y)
    n = len(x)

    cov_numer = sum([xi*yi for xi,yi in zip(x, y)]) - n * x_median * y_median
    cov_denom = n
    
    cov = cov_numer / cov_denom
    
    return round(cov, 2)

def get_hosted_num(table, service):
    """Gets the total amount of hosted movies for a specific service

        Args:
            table(MyPyTable): Table used to get the info desired
            service(str): used to denote the service we are getting the total amount of movies for

        Returns:
            movie_count: total number of movies hosted on the specified service
    """
    movie_count = 0
    col = MyPyTable.get_column(table, service, False)
    for val in col:
        if val == 1:
            movie_count += 1

    return movie_count

def total_occurrences(service, total):
    """Returns the percentage of movies a service has with respect to the total amount of movies accross all services

        Args:
            service(str): used to denote the service we are getting the total amount of movies for
            total(int/float): Total number of movies

        Returns:
            percent: Total perentage of movies a service has compared to the total
    """
    percent = service / total
    return percent * 100

def create_number_table(col_1, col_2, title_1, title_2):
    """Combines both IMDb rating and Rotten Tomatoes rating columns into one table

        Args:
            col_1([ints/flaots]): list of first column values
            col_2([ints/flaots]): list of secnf column values
            title_1(str): title for first column
            title_2(str): title for second column

        Returns:
            table: MyPyTable of combined columns passed in
    """
    new_data = []
    titles = []
    titles.append(title_1)
    titles.append(title_2)
    for i in range(len(col_1)):
        if col_1[i] != '':
            col_1[i] = col_1[i] * 10
    for i in range(len(col_2)):
        if col_2[i] != '':
            col_2[i] = int(col_2[i].rstrip('%'))
    for i in range(len(col_1)):
        new_data.append([col_1[i], col_2[i]])
    
    table = MyPyTable(titles, new_data)

    return table

def create_genre_table(col_1, col_2, title_1, title_2):
    """Combines a ratings column and the genre column to make a new table 

        Args:
            col_1([ints/flaots]): list of first column values
            col_2([ints/flaots]): list of secnf column values
            title_1(str): title for first column
            title_2(str): title for second column

        Returns:
            table: MyPyTable of combined columns passed in
    """
    new_data = []
    titles = []
    titles.append(title_1)
    titles.append(title_2)
    for i in range(len(col_1)):
        new_data.append([col_1[i], col_2[i]])
    
    table = MyPyTable(titles, new_data)
    return table

def find_unique_genres(val):
    """Finds all values in the column passed in

        Args:
            val([str]): column we will be traversing through to fins all unique str

        Returns:
            unique_genres: list of all unique genres in the genre column
    """
    string = ''
    unique_genre = []
    for genre in val:
        if string == '':
            string += genre
        else:
            string = string + ',' + genre
    new_list = string.split(',')
    for val in new_list:
        if val not in unique_genre:
            unique_genre.append(val)
    return unique_genre

def get_IMDb_rating(table, genre):
    """Gets all the IMDb ratings for a specifc genre

        Args:
            table(MyPyTable): Table used to get the info desired
            genre(str): used to denote the genre we are using to find IMDb ratings

        Returns:
            ratings: list of all the ratings for a specific genre
    """
    ratings = []
    genre_col = MyPyTable.get_column(table, "Genres", True)
    IMDb_col = MyPyTable.get_column(table, "IMDb", True)
    for i in range(len(genre_col)):
        if genre in genre_col[i]:
            ratings.append(IMDb_col[i])
    data_copy = copy.deepcopy(ratings)
    for val in ratings:
        if val == '':
            data_copy.remove(val)
    ratings = data_copy
    for i in range(len(ratings)):
        ratings[i] = ratings[i] * 10
    return ratings

def create_IMDb_list(table, genres):
    """Creates a dictionay to effeciently get all ratings for every unique genre

        Args:
            table(MyPyTable): Table used to get the info desired
            genres([str]): List of all unique genres

        Returns:
            plot_data: List of multiple list that will be used to plot the IMDb rating data
    """
    IMDb_dict = {}
    plot_data = []
    for val in genres:
        IMDb_dict[val] = []
    for key in IMDb_dict:
        ratings = get_IMDb_rating(table, key)
        IMDb_dict[key] = ratings
    for key in IMDb_dict:
        plot_data.append(IMDb_dict[key])
    
    return plot_data

def get_rt_rating(table, genre):
    """Gets all the Rotten Tomatoes ratings for a specifc genre

        Args:
            table(MyPyTable): Table used to get the info desired
            genre(str): used to denote the genre we are using to find Rotten Tomatoes ratings

        Returns:
            ratings: list of all the ratings for a specific genre
    """
    ratings = []
    genre_col = MyPyTable.get_column(table, "Genres", True)
    rt_col = MyPyTable.get_column(table, "Rotten Tomatoes", True)
    for i in range(len(genre_col)):
        if genre in genre_col[i]:
            ratings.append(rt_col[i])
    data_copy = copy.deepcopy(ratings)
    for val in ratings:
        if val == '':
            data_copy.remove(val)
    ratings = data_copy
    for i in range(len(ratings)):
        ratings[i] = int(ratings[i].rstrip('%'))
    return ratings

def create_rt_list(table, genres):
    """Creates a dictionay to effeciently get all ratings for every unique genre

        Args:
            table(MyPyTable): Table used to get the info desired
            genres([str]): List of all unique genres

        Returns:
            plot_data: List of multiple list that will be used to plot the Rotten Tomatoes rating data
    """
    rt_dict = {}
    plot_data = []
    for val in genres:
        rt_dict[val] = []
    for key in rt_dict:
        ratings = get_rt_rating(table, key)
        rt_dict[key] = ratings
    for key in rt_dict:
        plot_data.append(rt_dict[key])
    
    return plot_data