
import pandas as pd
import numpy as np
import itertools as it
import xlsxwriter
from collections.abc import Callable

class DataCore():


    """Main class for interacting with the data stored in the excel files


    ...
    Attributes
    ----
    excel_data : pd.Dataframe

    all the data from the excel file loaded into a pandas Dataframe


    """
    def load_excel(self,path : str) -> None:
        """Load the data from and excel file

        Failes if there is no excel file at the given path or
        if openpyxl is not installed"""
        #TODO remember that openpyxl must be installed for this to run
        #TODO verify loaded files
        excel_file = pd.ExcelFile(path)
        self.excel_data = {}
        for sheet_name in excel_file.sheet_names:
            #probably better to add regex search for wavelengt in sheet name
            self.excel_data[float(sheet_name)] = excel_file.parse(sheet_name,header=None)

    def create_excel_template(self,n_connectors,path="template.xlsx",n_wavelengths=1,n_fibers=1):


        #
        Sheet = np.zeros((n_connectors*n_connectors+2,n_fibers+2),dtype=object)

        #setting constant cells
        Sheet[0,0] = "Reference Configuration"
        Sheet[1,0] = "Reference Connector"
        Sheet[0,1] = ""
        Sheet[1,1] = "DUT"
        #setting numbering
        Sheet[1,2:n_fibers+2] = np.linspace(1,n_fibers,n_fibers)
        Sheet[2:n_connectors*n_connectors+2,1] = np.tile(np.linspace(1,n_connectors,n_connectors),n_connectors)
        Sheet[2:n_connectors*n_connectors+2,0] = np.repeat(np.linspace(1,n_connectors,n_connectors),n_connectors)


        excel_df = pd.DataFrame(Sheet)

        writer = pd.ExcelWriter(path,engine="xlsxwriter")
        for i in range(n_wavelengths):
            excel_df.to_excel(writer,sheet_name=f"wavelength_{i}",index=False,header=False)

        workbook = writer.book
        merge_format = workbook.add_format(
            {
                "bold": 1,
                "border": 1,
                "align": "center",
                "valign": "vcenter",
            }
        )
        for worksheet in workbook.worksheets():
            column_letter = xlsxwriter.utility.xl_col_to_name(n_fibers+2)

            print(f"C1:{column_letter}1")
            worksheet.set_column("A:B",20)
            worksheet.merge_range(f"C1:{column_letter}1", "Fiber Number",merge_format)
            for i in range(n_connectors):
                worksheet.merge_range(f"A{3+i*n_connectors}:A{2+(i+1)*n_connectors}", f"{i+1}",merge_format)

        writer.close()


    def all_cells(self,data : pd.DataFrame) -> np.ndarray:
        """Returns values all cells read from a given DataFrame

        Since data from the excel file is stored in a dictionary this should
        be called on output of DataCore.IL_wavelength"""
        return data.to_numpy()


    def all_IL_values(self,data : pd.DataFrame) -> np.ndarray:
        """"Returns values of all cells where we expect IL values"""
        return self.all_cells(data)[2:,2:]

    def n_connectors(self,data : pd.DataFrame) -> int:
        """Returns the expected number of connectors present in the data"""
        #the second dimensions corresponds to the number of connectors
        # the first one cannot be used since it contains multiple entries
        # for different wavelengths
        return int(np.sqrt(self.all_IL_values(data).shape[0]))

    def n_fibers(self,data : pd.DataFrame) -> int:
        """Returns the expected number of fiber present in the data"""
        #the second dimensions corresponds to the number of connectors
        # the first one cannot be used since it contains multiple entries
        # for different wavelengths
        return self.all_IL_values(data).shape[1]

    def n_jumpers(self,data : pd.DataFrame) -> int:
        """Returns the expected number of jumpers present in the data"""

        # there are two connectors per jumper
        return self.n_connectors(data)//2


    def wavelengths(self) -> list[float]:
        """Returns a list of wavelenghs found in the data"""

        return list(self.excel_data.keys())


    def IL_reference_connetor(self, data : pd.DataFrame, index : int) -> np.ndarray:
        """Returns values of IL for a given reference connector"""

        return self.all_IL_values(data)[:,index*self.n_connectors():(index+1)*self.n_connectors()]


    def IL_reference_jumper(self, data : pd.DataFrame, index : int) -> np.ndarray:
        """Returns values of IL for a given reference jumper"""
        return self.all_IL_values(data)[:,2*index*self.n_connectors():2*(index+1)*self.n_connectors()]



    def IL_wavelength(self,wavelength : float) -> pd.DataFrame:
        """Returns values of IL for a given wavelength"""


        return self.excel_data[wavelength]


    def filter_n_jumper(self,IL_data : pd.DataFrame, jumper_indecies : list[int]) -> np.ndarray:
        """Filters the data to only include the data for a given jumper indecies

        Works only on data of shape m^2 x n. In practive this means using data from
        DataCore.IL_wavelength
        """


        connector_indecies = []
        for jumper_index in jumper_indecies:
            connector_indecies.append(2*jumper_index)
            connector_indecies.append(2*jumper_index+1)

        excel_indecies = []
        for index_reference in connector_indecies:
            for index_dut in connector_indecies:
                excel_indecies.append(index_reference*self.n_connectors(IL_data)+index_dut)

        return self.all_IL_values(IL_data)[excel_indecies,:]


    def jumper_combinations(self, IL_data : pd.DataFrame, n_choices : int) -> list[np.ndarray]:

        if n_choices > self.n_jumpers(IL_data):
            print(f"Cannot chose {n_choices} from {self.n_jumpers}.")

        all_combinations_tupples = it.combinations(range(0,self.n_jumpers(IL_data)),n_choices)

        IL_data_combinations = []

        for combination in all_combinations_tupples:

            data = self.filter_n_jumper(IL_data,list(combination))

            IL_data_combinations.append(data)

        return IL_data_combinations

    def jumper_combinations_all_wavelengths(self, n_choices : int) -> dict[float, list[np.ndarray]]:


        wavelength_IL_combinations = {}
        for wavelength in self.wavelengths():

            IL_data_wavelength = self.IL_wavelength(wavelength)
            IL_data_combinations = self.jumper_combinations(IL_data_wavelength,n_choices)
            wavelength_IL_combinations[wavelength] = IL_data_combinations


        return wavelength_IL_combinations

    def map_dict(self, func, d : dict) -> dict:
        """Map a function over a dictionary

        For a dictionary {a : x, b : y, ...} and a function f
        this function returns {a : f(x), b : f(y), ...}"""
        return dict(map(lambda x : (x,func(d[x])),d))


    def IL_wavelengths(self) -> dict[float,np.ndarray]:
        return {wavelength : self.all_IL_values(self.IL_wavelength(wavelength)) for wavelength in self.wavelengths()}


    def IL_reference_connectors(self) ->  np.ndarray:
        """Create a list containg all IL values for each of the reference connectors
        
        Returns an array of shape n_connectors x n_wavelengths*n_connectors x n_fibers """
        wave_IL = self.excel_data
        connector_data = []
        for wave in wave_IL:
            data = wave_IL[wave]
            connector = np.split(self.all_IL_values(data),self.n_connectors(data),axis=0)
            connector_data.append(np.array(connector))
            
        return np.hstack(connector_data)
    
    def split_array(self,arr, k):
        split_arrays = [] 
        for i in range(k): 
            split_arrays.append(arr[i::k]) 
        return split_arrays
    
    def IL_dut_connectors(self) -> np.ndarray:
        """Create a list containg all IL values for each of the DUT connectors
        
        Returns an array of shape n_connectors x n_wavelengths*n_connectors x n_fibers.
        Should fullfill the same job as DataCore.IL_reference_connectors"""
        wave_IL = self.excel_data
        connector_data = []
        for wave in wave_IL:
            data = wave_IL[wave]
            connector = self.split_array(self.all_IL_values(data),self.n_connectors(data))
            connector_data.append(np.array(connector))
        print(np.hstack(connector_data).shape)

        return np.hstack(connector_data)
    
    def IL_fibers(self) -> np.ndarray:
        """Create a list containg all IL values for each of fiber
        
        Returns an array of shape n_fibers x n_wavelengths*n_connectors*n_connectors x 1.
        Could be flattened but it does not impact aggregate functions like mean, std, etc."""
        wave_IL = self.excel_data
        connector_data = []
        for wave in wave_IL:
            data = wave_IL[wave]
            connector = np.split(self.all_IL_values(data),self.n_fibers(data),axis=1)
            connector_data.append(np.array(connector))

        return np.hstack(connector_data)


    def filter_nan(self,A : np.ndarray) -> np.ndarray:
        """Remove NaN values from ndarray or list of ndarray"""
        if type(A) == type([]):
            float_cast = [ a.astype(float) for a in A]
            return np.array([ a[~np.isnan(a)] for a in float_cast])

        float_cast = A.astype(float)
        return float_cast[~np.isnan(float_cast)]


if __name__ == "__main__":

    #example
    DC = DataCore()
    DC.load_excel("ExampleData.xlsx")


    wavelength_ex = DC.wavelengths()[0]

    test_sheet = DC.IL_wavelength(wavelength_ex)

    print(f"Number of connectors {DC.n_connectors(test_sheet)}")
    print(f"Number of jumper {DC.n_jumpers(test_sheet)}")
    print(f"Number of fiber {DC.n_fibers(test_sheet)}")

    wave_combinations_IL_unfiltered = DC.jumper_combinations_all_wavelengths(4)
    print(wave_combinations_IL_unfiltered)
    wave_combinations_IL = DC.map_dict(DC.filter_nan, wave_combinations_IL_unfiltered)

    wave_combinations_IL_mean = DC.map_dict(lambda arr : np.mean(arr,axis=1), wave_combinations_IL)
    wave_combinations_IL_std = DC.map_dict(lambda arr : np.std(arr,axis=1), wave_combinations_IL)
    wave_combinations_IL_97th = DC.map_dict(lambda arr : np.percentile(arr,97,axis=1), wave_combinations_IL)


    print("mean,std and 97th percentile for the first 10 combinations of connectors\
        for all wavelengths")
    print(wave_combinations_IL_mean[1550][:10])
    print(wave_combinations_IL_std[1550][:10])
    print(wave_combinations_IL_97th[1550][:10])

    wave_IL_unfiltered = DC.IL_wavelengths()
    wave_IL = DC.map_dict(DC.filter_nan, wave_IL_unfiltered)

    wave_IL_mean = DC.map_dict(lambda arr : np.mean(arr,axis=0), wave_IL)
    wave_IL_std = DC.map_dict(lambda arr : np.std(arr,axis=0), wave_IL)
    wave_IL_97th = DC.map_dict(lambda arr : np.percentile(arr,97,axis=0), wave_IL)

    print("mean,std and 97th percentile for all wavelengths")
    print("\n",wave_IL_mean,wave_IL_std,wave_IL_97th)



    reference_connectors_IL_unfiltered = DC.IL_reference_connectors()
    reference_connectors_IL = list(map(DC.filter_nan,reference_connectors_IL_unfiltered))

    print("mean,std and 97th percentile for first 10 reference connectors")
    print(list(map(np.mean,reference_connectors_IL))[:10])
    print(list(map(np.std,reference_connectors_IL))[:10])
    print(list(map(lambda x : np.percentile(x,97),reference_connectors_IL))[:10])

    dut_connectors_IL_unfiltered = DC.IL_dut_connectors()
    dut_connectors_IL = list(map(DC.filter_nan,dut_connectors_IL_unfiltered))
    
    
    print("mean,std and 97th percentile for first 10 dut connectors")
    print(list(map(np.mean,dut_connectors_IL))[:10])
    print(list(map(np.std,dut_connectors_IL))[:10])
    print(list(map(lambda x : np.percentile(x,97),dut_connectors_IL))[:10])

    fibers_IL_unfiltered = DC.IL_fibers()
    fibers_IL = list(map(DC.filter_nan,fibers_IL_unfiltered))

    print("mean,std and 97th percentile for all fibers")
    print(list(map(np.mean,fibers_IL)))
    print(list(map(np.std,fibers_IL)))
    print(list(map(lambda x : np.percentile(x,97),fibers_IL)))
