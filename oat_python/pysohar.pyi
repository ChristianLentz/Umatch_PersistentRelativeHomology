from typing import List, Optional
from pandas import DataFrame



class FactoredBoundaryMatrixDowker:
    """
    The factored boundary matrix of a Dowker complex
    :param dowker_simplices a list of softed-in-ascending-order lists of integers
    :param max_homology_dimension the maximum dimension in which we want to compute homology
    """

    def __init__(self, dowker_simplices: List[List[int]], max_homology_dimension: int): 
        """
        Initializes a new FactoredBoundaryMatrixDowker instance with a list of (sorted) lists of integers.

        Args:
            dowker_simplices: A list of (sorted in strictly ascending order) lists of integers to use for initialization.
        """

    def homology(self) -> DataFrame:
        """
        Returns a Pandas DataFrame with information about homology, betti numbers, and cycle representatives
        """

    # def jordan_column_for_simplex( keymaj: List ) -> DataFrame:
    #     """
    #     Obtain a column of the Jordan basis associated with the U-match factorization

    #     :param keymaj: a list of integers
        
    #     :return L: a list of tuples `( s, a, b )`, where `s` is a simplex, and `a/b` is the coefficient of the simplex
    #     """


class FactoredBoundaryMatrixVr:
    """
    The factored boundary matrix of a filtered Vietoris Rips complex
    
    This object is uniquely determined by three user-defined parameters:
    - a dissimilarity_matrix
    - a maximum dissimilarity threshold
    - the maximum dimension in which we want to compute homology
    """

    def __init__(self,  dissimilarity_matrix: List[List[float]], dissimilarity_max: Optional[float],  homology_dimension_max:  int, ):             
        """
        Initializes a new FactoredBoundaryMatrixVr instance

        Args:
            dissimilarity_matrix: a symmetric matrix (such as a distance matrix) represented in list-of-list format
            dissimilarity_max: we only consttruct simplices with diameter diameter or smaller
            homology_dimension_max: the maximum dimension to compute homology
        """

    def homology(self) -> DataFrame:
        """
        Returns a Pandas DataFrame with information about homology, betti numbers, and cycle representatives
        """

    def optimize_cycle(self, birth_simplex: List[int] ) -> dict:
        """        
        Optimize a cycle representative

        Specifically, we employ the "edge loss" method to find a solution `x'` to the problem 

        `minimize Cost(Ax + z)`

        where 
            - `x` is unconstrained
        - `z` is a cycle representative for a (persistent) homology class associated to `birth_simplex`
        - `A` is a matrix composed of a subset of columns of the Jordna basis
        - `Cost(z)` is the sum of the absolute values of the products `z_s * diameter(s)`.

        # Arguments

        - The `birth_simplex` of a cycle represenative `z` for a bar `b` in persistent homology.
        - The `constraint` type for the problem. The optimization procedure works by adding linear
        combinations of column vectors from the Jordan basis matrix computed in the factorization.
        This argument controls which columns are available for the combination.
          - (default) **"preserve PH basis"** adds cycles which appear strictly before `birth_simplex`
            in the lexicographic ordering on filtered simplex (by filtration, then breaking ties by
            lexicographic order on simplices) and die no later than `birth_simplex`.  **Note** this is
            almost the same as the problem described in [Escolar and Hiraoka, Optimal Cycles for Persistent Homology Via Linear Programming](https://link.springer.com/chapter/10.1007/978-4-431-55420-2_5)
            except that we can include essential cycles, if `birth_simplex` represents an essential class. 
          - **"preserve PH basis (once)"** adds cycles which (i) are distince from the one we want to optimize, and
            (ii) appear (respectively, disappear) no later than the cycle of `birth_simplex`.  This is a looser
            requirement than "preserve PH basis", and may therefore produce a tighter cycle.  Note,
            however, that if we perform this optimization on two or more persistent homology classes in a
            basis of cycle representatives for persistent homology, then the result may not be a
            persistent homology basis.
          - **"preserve homology class"** adds every boundary vector
          - "preserve homology calss (once)" adds every cycle except the one represented by `birth_simplex`



        # Returns

        - The vectors `b`, `x`, and `y`
          - We separate `x` into two components: one made up of codimension-1 simplices (labeled "difference in bounding chains"), and one made up of codimension-0 simplices (labeled "difference in essential cycles")
        - The objective values of the initial and optimized cycles
        - The number of nonzero entries in the initial and optimized cycles
    
    # Related
    
    See
    
    - [Escolar and Hiraoka, Optimal Cycles for Persistent Homology Via Linear Programming](https://link.springer.com/chapter/10.1007/978-4-431-55420-2_5)
    - [Obayashi, Tightest representative cycle of a generator in persistent homology](https://epubs.siam.org/doi/10.1137/17M1159439)
    - [Minimal Cycle Representatives in Persistent Homology Using Linear Programming: An Empirical Study With User’s Guide](https://www.frontiersin.org/articles/10.3389/frai.2021.681117/full)
              # def jordan_column_for_simplex( keymaj: List ) -> DataFrame:
    #     
    """
    
    def persistent_relative_homology_lag_filtration(
        self, 
        delta: float, 
        return_cycle_representatives: bool, 
        return_bounding_chains: bool,
        subcomplex_filtration_max: Optional[float]
    ) -> DataFrame:
        """
        Returns a Pandas DataFrame with information about relative homology, betti numbers, and cycle representatives with the 
        assumption that the filtered subcomplex is identical the full complex with a constant lag/delay in the filtration. This 
        relative homology data frame is built from a U-match decomposition of the boundary matrix of the full complex over the 
        field of rational numbers. 
        
        # Arguments
        
        - `delta`:  a float indicating the lag/delay in the subcomplex filtrtion 
        - `return_cycle_representatives`: a boolean determining if cycle representatives are included in the returned data frame
        - `return_bounding_chains`: a boolean determining if bounding chains are included in the returned data frame
        - `subcomplex_filtration_max`: an optional parameter determining the maximal dissimilarity threeshold for subcomplex simplices. 
        If this parameter is ommitted, it defaults to positive infinty. 
        """
    
class FactoredBoundaryMatrixVrRelative:
    """
    The factored boundary matrix for a filtered pair of Vietoris-Rips Complexes, or equivalently, the factored boundary matrix of a filtered, 
    quotient chain complex. 
     
    We include methods for: 
    - persistent relative homology barcodes 
    - relative cycle representatives via basis matching
    - python wrappers for: 
         - sparse boundry matrices 
         - major and minor indices of sparse boundary matrices
         - sparse target (codomain) COMBs (containing relative boundary basis)
         - sparse source (domain) COMBs (containing relative cycle basis)
    """

    def __init__(
        self, 
        dissimilarity_matrix_full_space: List[List[float]], 
        dissimilarity_matrix_subspace: List[List[float]], 
        time_benchmarking: bool, 
        homology_dimension_max: Optional[int], 
        max_dissimilarity_threshold: Optional[float], 
        # subspace_indices_custom_filtration: Optional[List[float]] --> currently deprecated as unsafe
    ):             
        """
        Construct a pair of Vietoris-Rips complexes and factor the boundary matrix of the associated filtered, quotient chain complex over the field of rational numbers.  
         
        # Arguments
         
        - `dissimilarity_matrix_full_space`: a sparse dissimilarity matrix for a point cloud; missing entries will be treated as edges that never enter the filtration. 
        - `dissimilarity_matrix_subspace`: a sparse dissimilarity matrix for a point cloud which is a subset of the point cloud given by `dissimilarity_matrix_full_space`. 
        - `time_benchmarking`: a boolean determining if the oracle should keep track of time elapsed during construction, factorization, and related computations.
        - `homology_dimension_max`: the maximum dimension for which homology is desired. 
        - `max_dissimilarity_threshold`: the maximal diameter of simplices to be included.  
        
        # Deprecated (unsafe) Arguments 
        - `subspace_indices_custom_filtration`: an optional vector of filtration values mapping 0-simplices of the subcomplex to custom filtration values, allowing 
        the user customization and flexibility. 
        
        # Returns 
        
        - An instance of `oat_rust::src::algebra::chains::relative::RelativeBoundaryMatrixOracleWrapper`, which provides methods for filtered bases of relative cycles, 
        relative boundaries and relative homology. 
         
        # Panics
        
        Panics if: 
        - Provided dissimilarity matrices are not the same size. 
        - There exists a structural nonzero entry in the subspace dissimilarity matrix which is not present in the full space dissimilarity matrix.
        - Provided dissimilarity matrices are not symmetric. 
        - For either the subspace or the full space, there exists an edge with filtration parameter less than the filtration parameter of its vertices.
         
        These safety checks are performed by the constructors for 
        - `oat_rust::src::algebra::chains::relative::RelativeBoundaryMatrixOracle` 
        - `oat_rust::src::topology::simplicial::from::graph_Weights::ChainComplexVrFiltered`. 
        """

    def persistent_relative_homology(self, return_cycle_representatives: bool, return_bounding_chains: bool, trim_subspace_simplices_from_cycle_representatives: bool) -> DataFrame:
        """
        Extract a barcode and a basis of relative cycle representatives. 
        
        Computes persistent homology given the boundary matrix of a filtered, quotient chain complex. 
        In other words, a filtered basis of relative cycles mod relative boundaries.  
        
        - Edges of weight `>= dissimilarity_max` are excluded.
        - Relative homology is computed in dimensions 0 through `homology_dimension_max`, inclusive.
        
        # Arguments
        
        - `return_cycles_representatives`: a boolean determining if cycle representatives are included in the returned data frame. 
        - `return_bounding_chains`: a boolean determining if bounding chains are computed and returned when they exist. 
        - `trim_subcomplex_simplices_from_cycle_representatives`: a boolean determining if relative cycle representatives suppress all subspace simplices. 
        
        # Returns
         
        A Pandas data frame containing a persistent relative homology barcode and (possibly) cycle representatives.  
        """
        
    def persistent_relative_homology_time_benchmarking(self, return_cycle_representatives: bool, return_bounding_chains: bool) -> float:
        """
        Determine time required to a compute barcode and a basis of relative cycle representatives. 
        
        # Arguments
        
        - `return_cycles_representatives`: a boolean determining if cycle representatives are computed. 
        - `return_bounding_chains`: a boolean determining if bounding chains are computed when they exist.
        
        # Returns
         
        The time required to compute persistent relative homology, in seconds, as a 64-bit float. Will return 0 in the case that the user has not specified 
        `time_benchmarking` = `true` when initializing the oracle.
        """
        
    def row_indices_boundary_matrix(self) -> DataFrame:
        """ 
        Extract row indices for the boundary matrix of a filtered, quotient chain complex in sorted order.
         
        If the max homology dimension passed by the user when factoring the boundary matrix is `d`, then the indices include
         
        - every simplex of dimension `<= d`, and 
        - every simplex of dimension `d+1` that pairs with a simplex of dimension `d`.
        
        # Returns 
        
        - A Pandas data frame listing all simplices of the full complex totally ordered by (1) subcomplex membership (2) diameter/filtration and 
        (3) lexicographic ordering on vertices. Each row of the data frame includes a simplex, its diameter, and its subcomplex membership. 
        
        # Note: 
        
        - These are also the row indices of the matrices T, M, and S for the U-match of the boundary matrix. 
        """
    
    def column_indices_boundary_matrix(self) -> DataFrame:
        """
        Extract column indices for the boundary matrix of a filtered, quotient chain complex in sorted order.
         
        If the max homology dimension passed by the user when factoring the boundary matrix is `d`, then the indices include
         
        - every simplex of dimension `<= d`, and 
        - every simplex of dimension `d+1` that pairs with a simplex of dimension `d`.
        
        # Returns 
        
        - A Pandas data frame listing all simplices of the full complex totally ordered by (1) diameter/filtration and 
        (2) lexicographic ordering on vertices. Each row of the data frame includes a simplex and its diameter.
        
        # Note: 
        
        - These are also the row indices of the matrices T, M, and S for the U-match of the boundary matrix. 
        """

    def boundary_matrix(self) -> DataFrame: 
        """
        Returns the boundary matrix of a filtered, quotient chain complex formatted as a `scipy.sparse.csr_matrix` with rows/columns labeled by simplices.
        """
        
    def comb_source(self) -> DataFrame: 
        """
        Returns the source (domain) COMB for the U-match of the boundary matrix of a filtered, quotient chain complex formatted as a 
        `scipy.sparse.csr_matrix` with rows/column labeled by simplices.
        
        # Note: 
        
        - Row and column indices for this matrix may be retrieved, respectively, via the functions `row_indices_boundary_matrix` and `column_indices_boundary_matrix`.
        """
        
    def comb_target(self) -> DataFrame: 
        """
        Returns the target (codomain) COMB for the U-match of the boundary matrix of a filtered, quotient chain complex formatted as a 
        `scipy.sparse.csr_matrix` with rows/column labeled by simplices.
        
        # Note: 
        
        - Row and column indices for this matrix may be retrieved, respectively, via the functions `row_indices_boundary_matrix` and `column_indices_boundary_matrix`.
        """
        
    def index_matching(self) -> DataFrame:
        """
        Returns the index matching/matching matrix for the U-match of the boundary matrix of a filtered, quotient chain complex formatted as a 
        `scipy.sparse.csr_matrix` with rows/column labeled by simplices.
        
        # Note: 
        
        - Row and column indices for this matrix may be retrieved, respectively, via the functions `row_indices_boundary_matrix` and `column_indices_boundary_matrix`.
        """
        
    def matched_basis(self) -> DataFrame:
        """
        Returns the matched basis for the boundary matrix of a filtered, quotient chain complex. Specifically, this is a single matrix whose columns 
        contain bases for all relative cycles and all relative boundaires of the underlying quotient space, allowing for simplified extraction of 
        generators for relative homology classes. 
     
        # Note
        
        - The exported matched basis is formatted as a `scipy.sparse.csr_matrix` with rows/column labeled by simplices. 
        - Row and column indices for this matrix may be retrieved, respectively, via the functions `row_indices_boundary_matrix` and `column_indices_matched_basis`
        """
    
    def column_indices_matched_basis(self) -> DataFrame:
        """
        Returns the column indices of the matched basis for the boundary matrix of a filtered, quotient chain complex.
        """