import mysklearn.myutils as myutils
import copy
import csv
from os import truncate 
from tabulate import tabulate

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
        col_vals = []
        print(self.column_names)
        if self.data == []:
            return []
        # Check to see if col_idnetifier is a valid column name
        if isinstance(col_identifier, str):
            try:
                col_identifier = self.column_names.index(col_identifier)
            except ValueError:
                print(col_identifier, "is not a valid header name")
        # Check to see if col_idnetifier is a valid column index
        if isinstance(col_identifier, int):
            try:
                if col_identifier < len(self.column_names):
                    pass
            except ValueError:
                print(col_identifier, "is not a valid index")

        for row in self.data:
            if include_missing_values == True:
                col_vals.append(row[col_identifier])
            elif include_missing_values == False:
                if(row[col_identifier] != "NA" and row[col_identifier] != "N/A" and row[col_identifier] != ''):
                    col_vals.append(row[col_identifier])
        return col_vals

   
    def convert_to_numeric(self):
        """Try to convert each value in the table to a numeric type (float).

        Notes:
            Leave values as is that cannot be converted to numeric.
        """

        for row in self.data:
            for i in range(len(row)):
                try:
                    float_val = float(row[i])
                    row[i] = float_val
                except ValueError:
                   pass
        pass

 
    def drop_rows(self, rows_to_drop):
        """Remove rows from the table data.

        Args:
            rows_to_drop(list of list of obj): list of rows to remove from the table data.
        """
        for row in self.data:
            if row in rows_to_drop:
                self.data.remove(row)
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
            csv_reader = csv.reader(csv_file)
            for row in csv_reader:
                self.data.append(row)
            self.column_names = self.data[0]
            del self.data[0]
        self.convert_to_numeric()
        return self 

    def save_to_file(self, filename):
        """Save column names and data to a CSV file.

        Args:
            filename(str): relative path for the CSV file to save the contents to.

        Notes:
            Use the csv module.
        """
        with open(filename, 'w') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(self.column_names)
            csv_writer.writerows(self.data)
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
        dups = []
        seen = []
        indices = []
        
        for val in key_column_names:
            index = self.column_names.index(val)
            indices.append(index)

        for row in self.data:
            passed = True
            if seen == []:
                passed = False
            else:
                for r in seen:
                    passed = True
                    for i in indices:
                        if row[i] != r[i]:
                            passed = False
                    if passed:
                        break
            if passed == True:
                dups.append(row)
            else:
                seen.append(row)
        return dups

    def remove_rows_with_missing_values(self):
        """Remove rows from the table data that contain a missing value ("NA").
        """
        copy_data = copy.deepcopy(self.data)
        for row in self.data:
                if "NA" in row or '' in row:
                    copy_data.remove(row)
        self.data = copy_data
        pass

    def replace_missing_values_with_column_average(self, col_name):
        """For columns with continuous data, fill missing values in a column by the column's original average.

        Args:
            col_name(str): name of column to fill with the original average (of the column).
        """
        col_index = self.column_names.index(col_name)
        col_sum = 0
        num_col = 0
        for row in self.data:
            if row[col_index] != "NA":
                num_col += 1
                col_sum += row[col_index]
        col_avg = col_sum / num_col
        for row in self.data:
            if row[col_index] == "NA":
                row[col_index] = col_avg 
        pass

    def compute_summary_statistics(self, col_names):
        """Calculates summary stats for this MyPyTable and stores the stats in a new MyPyTable.

        Args:
            col_names(list of str): names of the continuous columns to compute summary stats for.

        Returns:
            MyPyTable: stores the summary stats computed. The column names and their order is as follows: ["attribute", "min", "max", "mid", "avg", "median"]
        """
        indices = []
        stats = []
        stats_title = ["attribute", "min", "max", "mid", "avg", "median"]
        if col_names == [] or self.data == []:
            pass
        else:
            for val in col_names:
                index = self.column_names.index(val)
                indices.append(index)
            
            for i in indices:
                temp = []
                col_values = []
                for row in self.data:
                    if row[i] != "NA" and isinstance(row[i], str) == False:
                        col_values.append(row[i])
                col_values.sort()
                index = len(col_values) // 2
                
                #compute median
                if len(col_values) % 2:
                    median = col_values[index]
                else:
                    median = sum(col_values[index - 1: index + 1]) / 2

                temp.append(self.column_names[i])
                temp.append(min(col_values))
                temp.append(max(col_values))
                temp.append((min(col_values) + max(col_values))/2)
                temp.append(sum(col_values)/len(col_values))
                temp.append(median)
                stats.append(temp)

        newTable = MyPyTable(stats_title, stats)
        return newTable

    def perform_inner_join(self, other_table, key_column_names):
        """Return a new MyPyTable that is this MyPyTable inner joined with other_table based on key_column_names.

        Args:
            other_table(MyPyTable): the second table to join this table with.
            key_column_names(list of str): column names to use as row keys.

        Returns:
            MyPyTable: the inner joined table.
        """
        new_column_names = []
        new_data = []

        right_column_names = copy.deepcopy(other_table.column_names)

        for name in key_column_names:
            right_column_names.remove(name)

        new_column_names = self.column_names + right_column_names

        for left_row in self.data:
            left_key = []
            for column in key_column_names:
                index = self.column_names.index(column)
                val = left_row[index]
                left_key.append(val)
            for right_row in other_table.data:
                right_key = []
                for column in key_column_names:
                    index = other_table.column_names.index(column)
                    val = right_row[index]
                    right_key.append(val)
                if left_key == right_key:
                    right_copy = copy.deepcopy(right_row)
                    for val in right_key:
                        right_copy.remove(val)
                    row_tba = left_row + right_copy
                    new_data.append(row_tba)
        
        return MyPyTable(new_column_names, new_data)

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
        new_column_names = []
        new_data = []
        indicies = []

        left_column_names = copy.deepcopy(self.column_names)
        right_column_names = copy.deepcopy(other_table.column_names)

        left_data = copy.deepcopy(self.data)

        for name in key_column_names:
            left_column_names.remove(name)
            right_column_names.remove(name)

        new_column_names = self.column_names + right_column_names

        for left_row in left_data:
            left_key = []
            match = False
            for name in key_column_names:
                index = self.column_names.index(name)
                val = left_row[index]
                left_key.append(val)
            match_index = 0
            for right_row in other_table.data:
                right_key = []
                for name in key_column_names:
                    index = other_table.column_names.index(name)
                    val = right_row[index]
                    right_key.append(val)

                if left_key == right_key:
                    indicies.append(match_index)
                    match = True
                    right_copy = copy.deepcopy(right_row)

                    for val in right_key:
                        right_copy.remove(val)
                    
                    row_tba = left_row + right_copy

                    new_data.append(row_tba)
                match_index += 1
            if (not match):
                row_tba = left_row

                for _ in right_column_names:
                    row_tba.append("NA")

                new_data.append(row_tba)
        na_indices = []
        populated_indices = []
        
        for i in range(len(new_column_names)):
            if not new_column_names[i] in other_table.column_names:
                na_indices.append(i)
            else:
                populated_indices.append(other_table.column_names.index(new_column_names[i]))
        
        right_data = copy.deepcopy(other_table.data)

        for row in right_data:
            for i in range(len(row)):
                if i not in populated_indices:
                    del row[i]
        
        for row in right_data:
            for i in na_indices:
                row.insert(i, "NA")

        for i in range(len(right_data)):
            if i not in indicies:
                new_data.append(right_data[i])

        return MyPyTable(new_column_names, new_data)