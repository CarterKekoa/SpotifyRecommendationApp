##############################################
# Programmer: Carter Mooring & Armando Valdez
# Class: CPCS 322-02, Spring 2021
# Spotify Api Project
# 4/18/21
# 
# 
# Description: This program opens and interprets a .csv file and stores its contents in a table.
#               The tables are then used to perform various task such as variable cleaning and specific value returns.
##############################################

import copy
import csv
from os.path import join
import statistics
from os import terminal_size, write 
from tabulate import tabulate # uncomment if you want to use the pretty_print() method
import mysklearn.myutils as myutils
# install tabulate with: pip install tabulate

# required functions/methods are noted with TODOs
# provided unit tests are in test_mypytable.py
# do not modify this class name, required function/method headers, or the unit tests
class MyPyTable:
    """Represents a 2D table of data with column names.

    Attributes:
        column_names(list of str): M column names
        data(list of list of obj): 2D data structure storing mixed type data. There are N rows by M columns.
    """

    def __init__(self, column_names=None, data=None):
        """Initializer for MyPyTable.

        Args:
            column_names(list of str): initial M column names (None if empty)
            data(list of list of obj): initial table data in shape NxM (None if empty)
        """
        if column_names is None:
            column_names = []
        self.column_names = copy.deepcopy(column_names)
        if data is None:
            data = []
        self.data = copy.deepcopy(data)

    def pretty_print(self):
        """Prints the table in a nicely formatted grid structure.
        """
        print(tabulate(self.data, headers=self.column_names))

    def pretty_print2(list):
        """Prints the a list in a nicely formatted grid structure.
        """
        for i in range(len(list)):
            print(list[i])

    def get_shape(self):
        """Computes the dimension of the table (N x M).

        Returns:
            int: number of rows in the table (N)
            int: number of cols in the table (M)
        """
        N = len(self.data)
        M = len(self.column_names)
        
        return N, M

    def get_column(self, col_identifier, include_missing_values=True):
        """Extracts a column from the table data as a list.

        Args:
            col_identifier(str or int): string for a column name or int
                for a column index
            include_missing_values(bool): True if missing values ("NA")
                should be included in the column, False otherwise.

        Returns:
            list of obj: 1D list of values in the column

        Notes:
            Raise ValueError on invalid col_identifier
        """
        # If false then first remove NA values before continuing
        if include_missing_values == False:
            MyPyTable.remove_rows_with_missing_values(self)
        
        # If col[i] is the column we are looking for, break to keep the i index location
        for i in range(len(self.column_names)):
            if self.column_names[i].lower() == col_identifier.lower():
                break
        
        single_col_list = []

        for row in self.data:
            single_col_list.append(row[i])

        return single_col_list

    def convert_to_numeric(self):
        """Try to convert each value in the table to a numeric type (float).

        Notes:
            Leave values as is that cannot be converted to numeric.
        """
        for row in self.data:
            for i in range(len(row)):
                try:
                    numeric_value = float(row[i])
                    row[i] = numeric_value
                except ValueError:
                    #print(row[i], "cannot be converted to a Float")
                    None
        pass

    def drop_rows(self, rows_to_drop):
        """Remove rows from the table data.

        Args:
            rows_to_drop(list of list of obj): list of rows to remove from the table data.
        """
        temp_table = copy.deepcopy(self.data)
        for row in self.data:
            for row2 in rows_to_drop:
                if row == row2:
                    temp_table.remove(row)
        
        self.data = temp_table
        pass 

    def load_from_file(self, filename):
        """Load column names and data from a CSV file.

        Args:
            filename(str): relative path for the CSV file to open and load the contents of.

        Returns:
            MyPyTable: return self so the caller can write code like: table = MyPyTable().load_from_file(fname)
        
        Notes:
            Use the csv module.
            First row of CSV file is assumed to be the header.
            Calls convert_to_numeric() after load
        """
        with open(filename) as csv_file:
            csv_read = csv.reader(csv_file)

            for row in csv_read:
                self.data.append(row)       #store values in object list, line by line
            
            self.column_names = self.data[0] # store first row (header) of table
            del self.data[0]                # delete the first row of the table
        
        self.convert_to_numeric()   

        return self 

    def save_to_file2(self, filename):
        """Save column names and data to a CSV file inside a folder.

        Args:
            filename(str): relative path for the CSV file to save the contents to.

        Notes:
            Use the csv module.
        """
        with open("./output_data/" + filename, 'w') as csv_file:
            csv_write = csv.writer(csv_file)
            csv_write.writerow(self.column_names)
            csv_write.writerows(self.data)
        pass

    def save_to_file(self, filename):
        """Save column names and data to a CSV file.

        Args:
            filename(str): relative path for the CSV file to save the contents to.

        Notes:
            Use the csv module.
        """
        with open(filename, 'w') as csv_file:
            csv_write = csv.writer(csv_file)
            csv_write.writerow(self.column_names)
            csv_write.writerows(self.data)
        pass

    def find_duplicates(self, key_column_names):
        """Returns a list of duplicates. Rows are identified uniquely baed on key_column_names.

        Args:
            key_column_names(list of str): column names to use as row keys.

        Returns: 
            list of list of obj: list of duplicate rows found
        
        Notes:
            Subsequent occurrence(s) of a row are considered the duplicate(s). The first instance is not
                considered a duplicate.
        """
        total_dups = []
        seen = []
        header_range = []

        for head in key_column_names:
            header_range.append(self.column_names.index(head))
        
        for row in self.data:
            passed = True
            if seen == []:
                passed = False
            else:
                for val in seen:
                    passed = True
                    for head in header_range:
                        if row[head] != val[head]:
                            passed = False
                    
                    if passed:
                        break

            if passed:
                total_dups.append(row)
            elif seen.append(row):
                seen.append(row)
        
        return total_dups
    


    def remove_rows_with_missing_values(self):
        """Remove rows from the table data that contain a missing value ("NA").
        """
        table_deep_copy = copy.deepcopy(self.data)
        
        for row in self.data:
            for i in range(len(row)):
                if row[i] == "NA" or row[i] == '':
                    if row in table_deep_copy:
                        table_deep_copy.remove(row)
        
        self.data = table_deep_copy         # Store table with no NA values back into self
        pass

    def replace_missing_values_with_column_average(self, col_name):
        """For columns with continuous data, fill missing values in a column by the column's original average.

        Args:
            col_name(str): name of column to fill with the original average (of the column).
        """
        row_list = []
        avg = 0
        count = 0
        row_count = 0

        # If col[i] is the column we are looking for, break to keep the i index location
        for i in range(len(self.column_names)):
            if self.column_names[i].lower() == col_name.lower():
                break
        
        for row in self.data:
            row_count += 1
            if row[i] == "NA":  
                row_list.append(row_count)      # Mark the row location to be changed later
            else:
                # prep values to calculate the avg once finished iterating
                count += 1
                avg = avg + row[i]
        
        avg = avg / count

        # Iterate through list of rows with missing values and change them to avg values
        for NAs in row_list:
            self.data[NAs - 1][i] = avg
        
        pass 

    def compute_summary_statistics(self, col_names):
        """Calculates summary stats for this MyPyTable and stores the stats in a new MyPyTable.

        Args:
            col_names(list of str): names of the continuous columns to compute summary stats for.

        Returns:
            MyPyTable: stores the summary stats computed.
        """
        stats = []
        header = ["attribute", "min", "max", "mid", "avg", "median"]
        # iterate the col_names list given
        for j in range(len(col_names)):
            # iterate through headers
            for i in range(len(self.column_names)):
                # If col[i] is the column we are looking for, break to keep the i index location
                if self.column_names[i].lower() == col_names[j].lower():
                    col_list = []
                    stat_results = []
                    
                    # store all of header i's values in their own list
                    for row in self.data:
                        if(row[i] != "NA"):
                            col_list.append(float(row[i]))
                    
                    # check to see if the col_list given is empty
                    if col_list != []:
                        stat_results.append(self.column_names[i])   # append the column header name

                        minimum = min(col_list)                     # calcualte the minimum
                        stat_results.append(minimum)
                        
                        maximum = max(col_list)                     # calculate the maximum
                        stat_results.append(maximum)
                        
                        mid = ((minimum + maximum) / 2)             # calculate the mid (half way between min and max),
                        stat_results.append(mid)

                        mean = statistics.mean(col_list)            # calculate the avg
                        stat_results.append(mean)

                        median = statistics.median(col_list)        # calculate the median
                        stat_results.append(median)
                    else:
                        return MyPyTable(header, [])                # list was empty so return empty list
            stats.append(stat_results)                              # append the statistics list for each column requested 
        return MyPyTable(header, stats)

    def perform_inner_join(self, other_table, key_column_names):
        """Return a new MyPyTable that is this MyPyTable inner joined with other_table based on key_column_names.

        Args:
            other_table(MyPyTable): the second table to join this table with.
            key_column_names(list of str): column names to use as row keys.

        Returns:
            MyPyTable: the inner joined table.
        """
        header = (self.column_names + list(set(other_table.column_names) - set(self.column_names)))     # stores the new joined header that will be returned after join
        inner_table = []       # stores the joined rows on a match
        key_header_indexes = [] # trackers the indexes of  the key_col_names compared to the tables header locations 

        # iterate through the key_cols_names given and find their locations in each table
        for keys in key_column_names:
            # Grab the column positions in both tables for the key value
            for i in range(len(self.column_names)):
                if self.column_names[i].lower() == keys.lower():
                    break
            for j in range(len(other_table.column_names)):
                if other_table.column_names[j].lower() == keys.lower():
                    break
            
            # store header index locations in a list
            temp = []
            temp.append(i)
            temp.append(j)
            key_header_indexes.append(temp)
            
        # iterate through the first tables rows
        for row in self.data:
            # iterate through the second tables rows
            for row2 in other_table.data:
                indexes = key_header_indexes[0]            # grab the first key_col_name index locations
                # if match for first key_col_name index locations
                if row[indexes[0]] == row2[indexes[1]]:
                    # Now check if we were given 2 key_col_name's
                    if len(key_column_names) > 1:
                        # given 2 headers so now grab the second key_col_name index locations
                        indexes2 = key_header_indexes[1]
                        # check if the second key_col_name index is also a match
                        if row[indexes2[0]] == row2[indexes2[1]]:
                            # Match in both header index loactions in both tables so append 
                            inner_table.append((row + list(set(row2) - set(row))))
                    else:
                        # just one header to check so easy append
                        inner_table.append((row + list(set(row2) - set(row))))

        return MyPyTable(header, inner_table)

    def perform_full_outer_join(self, other_table, key_column_names):
        """Return a new MyPyTable that is this MyPyTable fully outer joined with other_table based on key_column_names.

        Args:
            other_table(MyPyTable): the second table to join this table with.
            key_column_names(list of str): column names to use as row keys.

        Returns:
            MyPyTable: the fully outer joined table.

        Notes:
            Pad the attributes with missing values with "NA".
        """
        new_header = []
        cols1 = []
        cols2 = []
        keys1 = []
        keys2 = []
        joined_table = []

        # iterate through both tables column names to append to join header
        for val in self.column_names:
            new_header.append(val)
        for val in other_table.column_names:
            if val not in new_header:
                new_header.append(val)
        
        
        # for key column names passed in, grab indexes in each table
        for val in key_column_names:
            cols1.append(self.column_names.index(val))
            cols2.append(other_table.column_names.index(val))
        
        
        # iterate through all of self, append indexes
        for i in range(len(self.data)):
            curr = []
            for j in cols1:
                curr.append(self.data[i][j])
            keys1.append(curr)
        
        # iterate through all of other, append indexes
        for i in range(len(other_table.data)):
            curr = []
            for j in cols2:
                curr.append(other_table.data[i][j])
            keys2.append(curr)
        
        num_extra_rows = len(new_header) - len(self.column_names)       # grab amount of extra rows left

        # for amount of keys 
        for i in range(len(keys1)):
            current = copy.deepcopy(self.data[i])
            # key match so join rows
            if keys1[i] in keys2:
                for val in other_table.column_names:
                    if val not in self.column_names:
                        current.append(other_table.data[keys2.index(keys1[i])][other_table.column_names.index(val)]) # doesnt work since i corresponds to first table  
            else: 
                # keys don't match add in row with 'NA's
                for i in range(num_extra_rows):
                    current.append('NA')
            joined_table.append(current)

        # Now for any missing rows, add them to table
        for i in range(len(keys2)):
            # not matching keys arnt added, do here
            if keys2[i] not in keys1: 
                current = []
                for col_name in new_header:
                    if col_name in other_table.column_names: 
                        current.append(other_table.data[i][other_table.column_names.index(col_name)])
                    else:
                        current.append('NA')  
                joined_table.append(current)

        return MyPyTable(new_header, joined_table)