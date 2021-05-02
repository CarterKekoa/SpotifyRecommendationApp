"""
Programmer: Armando Valdez
Class: CptS 322-02, Spring 2021
Programming Assignment #3
2/25/21

Description: This program holds reusable plotting functions
"""
import matplotlib.pyplot as plt
import utils
def frequency_chart(x, y, title, xlabel, ylabel):
    """Methodd used to produce frequency charts

        Args:
            x([values]): x plot values
            y([values]): y plot values
            title(str): title of chart
            xlabely(str): x-axis chart label
            label(str): y-axis chart label

        Returns:
            A frequency chart
    """
    new_x = list(range(len(x)))
    plt.figure(figsize=(15,10))
    plt.bar(new_x, y)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(color="Gray", linestyle='--', linewidth=1, axis='y', alpha=0.7)
    plt.grid(color='Gray', linestyle='--', linewidth=1, axis='x', alpha=0.7)
    xtick_labels = x
    plt.xticks(new_x, xtick_labels, rotation=45, horizontalalignment="right")
    plt.tick_params(axis='x', which='major', labelsize=10)
    plt.show()

def modified_hist(x, y, title, xlabel, ylabel):
    """Methodd used to produce a bar chart as a histogram

        Args:
            x([values]): x plot values
            y([values]): y plot values
            title(str): title of chart
            xlabely(str): x-axis chart label
            label(str): y-axis chart label

        Returns:
            A bar chart modified to look like a histogram
    """
    plt.figure()
    plt.bar(x[:-1], y, width=x[1] - x[0], align='edge')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tick_params(axis='x', which='major', labelsize=10)
    plt.xticks(None, None, rotation='45', horizontalalignment="right")
    plt.show()

def percent_global_sales(x, y, title):
    """Methodd used to produce a Pie chart

        Args:
            x([values]): pie table titles
            y([values]): pie table percentages
            title(str): title of chart

        Returns:
            A Pie chart
    """
    plt.figure()
    plt.title(title)
    plt.pie(y, labels=x, autopct="%1.1f%%")
    plt.show()

def plot_hist(vals, bins, title, xlabel, ylabel):
    """Methodd used to produce a histogram

        Args:
            vals([int/floats]): x plot values
            bins([floats]): y plot values
            title(str): title of chart
            xlabely(str): x-axis chart label
            label(str): y-axis chart label

        Returns:
            A histogram
    """
    plt.hist(vals, bins)
    plt.xticks(None, None, rotation=45)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(None, None, rotation='45', horizontalalignment="right")
    plt.show()

def plot_scatter(x, y, title, xlabel, ylabel):
    """Methodd used to produce a scatter plot

        Args:
            x([values]): x plot values
            y([values]): y plot values
            title(str): title of chart
            xlabely(str): x-axis chart label
            label(str): y-axis chart label

        Returns:
            A scatter plot
    """
    plt.scatter(x, y)
    m, b = utils.calc_regression(x, y)
    print("best fit line: y =", round(b, 4) ,"+ (",round(m, 4),")x")
    regression = [b + m * xi for xi in x]
    corr = utils.calc_correlation_coeficient(x, y)
    cov = utils.calc_covariance(x, y) 
    plt.plot(x, regression, color='red', label="Corr:"+str(corr)+"\nCov:"+str(cov)+"")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.show()

def plot_boxplot(data, tick_labels, title, xlabel, ylabel):
    """Methodd used to produce a boxplot

        Args:
            data([int/floats]): x plot values
            tick_labels([Str]): y plot values
            title(str): title of chart
            xlabely(str): x-axis chart label
            label(str): y-axis chart label

        Returns:
            A boxplot
    """
    plt.figure(figsize=(20,10))
    plt.boxplot(data)
    ticks = []
    for i in range(len(tick_labels)):
        ticks.append(i + 1)
    plt.xticks(ticks, tick_labels, rotation=45, horizontalalignment="right")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show



