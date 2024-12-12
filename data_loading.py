
import pandas as pd
import numpy as np
import itertools as it
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
        #TODO switch the columns and change the excel file templete
        #TODO remember that openpyxl must be installed for this to run
        #TODO verify loaded files
        self.excel_data = pd.read_excel(path,usecols="G:AG",engine="openpyxl")

    def all_cells(self) -> np.ndarray:
        """Returns values all cells read from the excel file"""
        return self.excel_data.to_numpy()


    def all_IL_values(self) -> np.ndarray:
        """"Returns values of all cells where we expect IL values"""
        # bounds [3:,1;] correspond to the main portion of the data in the excel file
        # everything else is data about wavelengths and jumper/connectors indecies
        return self.all_cells()[3:,1:]

    def n_connectors(self) -> int:
        """Returns the expected number of connectors present in the data"""
        #the second dimensions corresponds to the number of connectors
        # the first one cannot be used since it contains multiple entries
        # for different wavelengths
        return self.all_IL_values().shape[1]

    def n_fibers(self) -> int:
        """Returns the expected number of fibers present in the data"""

        # there are two connectors per fiber
        return self.n_connectors()//2


    def wavelengths(self) -> list[float]:
        """Returns a list of wavelenghs found in the data"""
        wavelength_column = self.all_cells()[3:,0]

        return list(filter(lambda x : not np.isnan(x),set(wavelength_column)))


    def IL_connetor(self, index : int) -> np.ndarray:
        """Returns values of IL for a given connector (column)"""

        return self.all_IL_values()[:,index]


    def IL_fiber(self, index : int) -> np.ndarray:
        """Returns values of IL for a given fiber (column)"""
        return self.all_IL_values()[:,[2*index, 2*index + 1]]


    def IL_wavelength(self,wavelength : float) -> np.ndarray:
        """Returns values of IL for a given wavelength"""

        wavelength_mask = self.all_cells()[3:,0] == wavelength

        return self.all_IL_values()[wavelength_mask]


    def filter_n_fibers(self,IL_data : np.ndarray, fiber_indecies : list[int]) -> np.ndarray:
        """Filters the data to only include the data for a given fiber indecies

        Works only on data of shape m x m. In practive this means using data from
        DataCore.IL_wavelength
        """

        if IL_data.shape[0] != IL_data.shape[1]:
            print(f"Fitering data by fibers will most likely not work since the data has\
                dimension 0 - {IL_data.shape[0]} not equal to dimension 1 - {IL_data.shape[1]}.\
                This function works on square data matrices. Consider filtering by wavelength first.")


        connector_indecies = []
        for fiber_index in fiber_indecies:
            connector_indecies.append(2*fiber_index)
            connector_indecies.append(2*fiber_index+1)

        return IL_data[connector_indecies,:][:,connector_indecies]


    def fiber_combinations(self, IL_data : np.ndarray, n_choices : int) -> list[np.ndarray]:

        if n_choices > self.n_fibers():
            print(f"Cannot chose {n_choices} from {self.n_fibers}.")

        all_combinations_tupples = it.combinations(range(0,self.n_fibers()),n_choices)

        IL_data_combinations = []

        for combination in all_combinations_tupples:

            data = self.filter_n_fibers(IL_data,list(combination))

            IL_data_combinations.append(data)

        return IL_data_combinations

    def fiber_combinations_all_wavelengths(self, n_choices : int) -> dict[float, list[np.ndarray]]:


        wavelength_IL_combinations = {}
        for wavelength in self.wavelengths():

            IL_data_wavelength = self.IL_wavelength(wavelength)
            IL_data_combinations = self.fiber_combinations(IL_data_wavelength,n_choices)
            wavelength_IL_combinations[wavelength] = IL_data_combinations


        return wavelength_IL_combinations

    def map_dict(self, func, d : dict) -> dict:
        """Map a function over a dictionary

        For a dictionary {a : x, b : y, ...} and a function f
        this function returns """
        return dict(map(lambda x : (x,func(d[x])),d))


    def IL_wavelengths(self) -> dict[float,np.ndarray]:
        return {wavelength : self.IL_wavelength(wavelength) for wavelength in self.wavelengths()}


    def IL_connectors(self) -> list[np.ndarray]:
        return np.split(self.all_IL_values(),self.n_connectors(),axis=1)


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
    DC.load_excel("RM example data.xlsx")

    wave_combinations_IL_unfiltered = DC.fiber_combinations_all_wavelengths(10)
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



    connectors_IL_unfiltered = DC.IL_connectors()
    connectors_IL = list(map(DC.filter_nan,connectors_IL_unfiltered))

    print("mean,std and 97th percentile for first 10 connectors")
    print(list(map(np.mean,connectors_IL))[:10])
    print(list(map(np.std,connectors_IL))[:10])
