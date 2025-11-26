//! 
//! Boundary oracle of a filtered, quotient chain complex from a filtered pair of Vietoris-Rips complexs. Includes traits necessary for 
//! U-Match Decomposition and functionality for exact computation of filtered bases for relative cycles, relative boundaries and relative 
//! homology. 
//! 

use std::clone::Clone;
use std::fmt::Debug;
use std::hash::Hash;
use std::sync::Arc;
use itertools::Itertools;

use crate::algebra::chains::relative::traits::{FactorFromArc, VariableSortOracleKeys, SimplexDiameter}; 
use crate::algebra::chains::relative::order::{
    HashMapOrderOperator, OrderOperatorFullComplexFiltrationSimplices, RelativeBoundaryMatrixRowIndexOrderOperator, SourceColumnIndexOrderOperator, TargetColumnIndexOrderOperator
}; 
use crate::algebra::rings::operator_traits::{Semiring, Ring, DivisionRing};
use crate::topology::simplicial::simplices::filtered::SimplexFiltered;
use crate::utilities::order::{JudgePartialOrder, OrderOperatorByKeyCutsom}; 
use crate::algebra::matrices::types::matching::GeneralizedMatchingArrayWithMajorOrdinals;
use crate::algebra::matrices::query::{IndicesAndCoefficients, MatrixEntry, ViewColDescend, ViewRowAscend};
use crate::algebra::matrices::operations::umatch::row_major::{comb::{CombCodomainInv, CombDomain, CombCodomain}, Umatch};
use crate::algebra::matrices::operations::multiply::{BimajorProductMatrix, BimajorSparseProductMatrix};
use crate::algebra::vectors::entries::{KeyValGet, KeyValNew, KeyValSet}; 
use crate::topology::simplicial::from::graph_weighted::{ChainComplexVrFiltered, LazyOrderedCoboundary}; 

//  ===========================================================
//  Struct: RelativeBoundaryMatrixOracle
//  ===========================================================

#[derive(Clone, Debug)]
/// 
/// A lazy boundary oracle of a filtered, quotient chain complex modeled off of [`ChainComplexVrFiltered`], and constructed from 
/// a pair of dissimilarity matrices. 
/// 
/// This oracle is constructed from two point clouds provided as dissimilarity matrices, one of which is a subset of the other. 
/// The dissimilarity matrix for the superset data is used to construct a `ChainComplexVrFiltered` struct, which can be used to 
/// construct filtered bases for cycles, boundaries and homology. The subset dissimilarity matrix is used for the construction 
/// of custom order operators which are used to modfify the `ChainComplexVrFiltered` oracle, thus allowing for the construction 
/// of bases for relative cycles, relative boundaries and relative homolgy via U-Match Decomposition. 
/// 
/// Functionality for computing relative cycles, relative boundaries and relative homology is not included in this struct. 
/// See [`RelativeBoundaryMatrixOracleWrapper`]. 
/// 
pub struct RelativeBoundaryMatrixOracle<DissimilarityMatrix, Filtration, Coefficient, RingOperator, OrderOperatorSubComplex>
    where 
        DissimilarityMatrix: IndicesAndCoefficients<ColIndex=usize, RowIndex=usize, Coefficient=Filtration> + ViewRowAscend + MatrixEntry + Clone + Debug,
        DissimilarityMatrix::EntryMajor: KeyValGet<usize, Filtration> + Debug + Copy,
        Filtration: Clone + Copy + Debug + PartialOrd + Ord + Hash,
        Coefficient: Clone + Debug,
        RingOperator: Semiring<Coefficient> + Ring<Coefficient> + DivisionRing<Coefficient> + Copy + Debug, 
        OrderOperatorSubComplex: JudgePartialOrder<SimplexFiltered<Filtration>> + SimplexDiameter<Filtration> + Clone
{  
    /// Dissimilarity matrix for full space data.
    pub full_space_data: DissimilarityMatrix,
    /// Ring operator for boundary matrices of both VR complexes. 
    pub ring_operator: RingOperator, 
    /// Maximal dimension of simplices to include (for sake of time complexity, should be no more than 2 or 3).
    pub max_simplex_dimension: usize, 
    /// Vietoris-Rips filtration / filtered chain complex (full space data)   
    pub chain_complex_vr_full_space: ChainComplexVrFiltered<DissimilarityMatrix, Filtration, Coefficient, RingOperator>, 
    /// Order operator: rows (major keys) of `RelativeBoundaryMatrixOracle`
    pub row_index_order_operator: RelativeBoundaryMatrixRowIndexOrderOperator<Filtration, OrderOperatorFullComplexFiltrationSimplices<DissimilarityMatrix, Filtration>, OrderOperatorSubComplex>,
    /// Order operator: columns (minor keys) of `RelativeBoundaryMatrixOracle`
    pub column_index_order_operator: OrderOperatorFullComplexFiltrationSimplices<DissimilarityMatrix, Filtration>, 
    /// Order opearator: simplices in the sub complex filtration. Needed for constructing filtrations of relative cycles and boundaries.
    pub subspace_filtration_order_operator: OrderOperatorSubComplex // OrderOperatorSubComplexFiltrationSimplices<DissimilarityMatrix, Filtration>,
}    

/// Implementation and methods for `RelativeBoundaryMatrixOracle`
impl <'a, DissimilarityMatrix, Filtration, Coefficient, RingOperator, OrderOperatorSubComplex>
    RelativeBoundaryMatrixOracle<DissimilarityMatrix, Filtration, Coefficient, RingOperator, OrderOperatorSubComplex>
        where
            DissimilarityMatrix: IndicesAndCoefficients<ColIndex=usize, RowIndex=usize, Coefficient=Filtration> + ViewRowAscend + MatrixEntry + Debug + Clone,
            DissimilarityMatrix::EntryMajor: KeyValGet<usize, Filtration> + Debug + Copy,
            Filtration: Clone + Copy + Debug + PartialOrd + Ord + Hash,
            Coefficient: Clone + Debug + Hash,
            RingOperator: Semiring<Coefficient> + Ring<Coefficient> + DivisionRing<Coefficient> + Copy + Debug, 
            OrderOperatorSubComplex: JudgePartialOrder<SimplexFiltered<Filtration>> + SimplexDiameter<Filtration> + Clone
{      
    /// 
    /// Construct a `RelativeBoundaryMatrixOracle` structure. 
    /// 
    /// - `data_full_space` is a dissimilarity matrix for the data points in the full space. This is used to construct the 
    /// underlying/wrapped `ChainComplexVrFiltered` boundary matrix oracle. 
    /// - `data_subspace` is a dissimilarity matrix for the data points in the subspace. In other words, each point of 
    /// `data_subspace` must be present in `data_full_space`. This is used to construct a custom order operator which 
    /// places a total order on the major keys of the `RelativeBoundaryMatrixOracle`.
    /// - `data_full_space_size` is the number of rows (respectively, columns) of `data_full_space`.
    /// - `data_subspace_size` is the number of rows (respectively, columns) of `data_subspace`. 
    /// - `dissimilarity_max` is the maximum dissimilarity threshold. 
    /// - `dissimilarity_min` is the minimum dissimilarity threshold. 
    /// - `ring_op` is the operator for the coefficient ring used to construct the matrix oracle. 
    /// - `subspace_order_operator` is an order operator comparing simplices by their birth in the subcomplex.
    /// - `max_dim` is the maximal simplex dimension for simplices in both `full_space_vr` and `subspce_vr`. For the sake 
    /// of time and space complexity, this should be no more than 2 or 3 in practice. The maximal homology dimension that 
    /// can be computed is this `max_dim` - 1. 
    /// 
    /// ## Safety Checks
    /// 
    /// - Ensure both provided dissimilarity matrices are the same size. 
    /// - Ensure that each nonzero entry in the subspace dissimilarity matrix is also present in the full space dissimilarity matrix.
    /// - Ensure both dissimilairty matrices are symmetric. 
    /// - Ensure that, for both the subspace and the full space, that the filtration parameter of every edge is greater than or equal 
    /// to the filtration parameter of its vertices.
    /// 
    /// The size of the dissimilarity matrices (number of rows and columns) is given by the number of points in the full space 
    /// data. Thus, the first condition above states that the dissimilarity matrix provided for the subspace data should be more 
    /// sparse than the dissimilarity matrix provided for the full space data. The second condition above states that the provided 
    /// subspace data is actually a subspace of the full space.  
    /// 
    /// The last two conditions are checked by the constructor for `ChainComplexVrFiltered`, and thus unit testing for this struct 
    /// does not include these test cases. 
    /// 
    pub fn new(
        data_full_space: DissimilarityMatrix,
        data_subspace: DissimilarityMatrix,
        data_full_space_size: usize, 
        data_subspace_size: usize, 
        dissimilarity_max: Filtration,
        dissimilarity_min: Filtration,
        ring_op: RingOperator,
        subspace_order_operator: OrderOperatorSubComplex, 
        max_dim: usize
    ) -> Self

    {   
        // Ensure both provided dissimilarity matrices are the same size.
        if data_full_space_size != data_subspace_size { 
            panic!("\n\nError: Constructing `RelativeBoundaryMatrixOracle` failed. Dissimilarity matrices passed to constructor are not the same size. Note this condition is checked via user provided data. Ensure provided dissimilarity matrices are actually the same size, and that these parameters are computed correctly. \n This message is generated by OAT.\n\n");
        }
        // Ensure that each structural nonzero entry in `data_subspace` is also present in `data_full_space`.
        let mut num_subspace_entries = 0; 
        for i in 0..data_subspace_size { 
            for entry in data_subspace.view_major_ascend(i) {
                let j = entry.key();
                // get entry (i,j) of both dissimilarity matrices 
                let subspace_matrix_entry = entry.val(); 
                let full_space_matrix_entry = data_full_space.entry_major_at_minor(i, j);
                // check cases in which the entries are not identical 
                if full_space_matrix_entry.is_some() { 
                    if subspace_matrix_entry != full_space_matrix_entry.unwrap() { 
                        panic!("\n\nError: Constructing `RelativeBoundaryMatrixOracle` failed. Entry ({:?},{:?}) of each provided dissimilarity matrix is present, but they are not identical; subspace relationship not satisfied. \n This message is generated by OAT.\n\n", i, entry.key());
                    }
                } 
                else { 
                    panic!("\n\nError: Constructing `RelativeBoundaryMatrixOracle` failed. Entry ({:?},{:?}) of full dissimilarity matrix is `None`, and the corresponding entry of subspace dissimilarity matrix is `Some`; subspace relationship not satisfied. \n This message is generated by OAT.\n\n", i, entry.key());
                }
                num_subspace_entries = num_subspace_entries + 1; 
            }
        }
        // Ensure that `data_subspace` encodes a point cloud which is a proper subset of the full space. 
        // In any other case, relative homology is trivial, or equivalent to absolute homology! 
        if num_subspace_entries == 0 || num_subspace_entries == 1 { 
            panic!("\n\nError: Constructing `RelativeBoundaryMatrixOracle` failed. The provided subspace data is an empty set or a single point, and thus relative homology is trivial. \n This message is generated by OAT.\n\n");
        }
        // NOTE: this case does not need to be checked!
        // if num_subspace_entries == data_full_space_size * data_full_space_size { 
        //     panic!("\n\nError: Constructing `RelativeBoundaryMatrixOracle` failed. The provided subspace data is equivalent to the full space data, and thus relative homology is trivial. \n This message is generated by OAT.\n\n");
        // }
        // construct oracle and return 
        let full_space_vr = ChainComplexVrFiltered::new(
            data_full_space.clone(),
            data_full_space_size,
            dissimilarity_max,
            dissimilarity_min,
            ring_op
        );
        let column_order_operator = OrderOperatorFullComplexFiltrationSimplices::new(
            data_full_space.clone(), 
            dissimilarity_min
        );
        let row_order_operator = RelativeBoundaryMatrixRowIndexOrderOperator::new( 
            column_order_operator.clone(), 
            subspace_order_operator.clone()
        );
        RelativeBoundaryMatrixOracle { 
            full_space_data: data_full_space.clone(),
            ring_operator: ring_op,
            max_simplex_dimension: max_dim, 
            chain_complex_vr_full_space: full_space_vr, 
            row_index_order_operator: row_order_operator,
            column_index_order_operator: column_order_operator, 
            subspace_filtration_order_operator: subspace_order_operator
        }
    }
}

/// Implement IndicesAndCoefficients for `RelativeBoundaryMatrixOracle`
impl<'a, DissimilarityMatrix, Filtration, Coefficient, RingOperator, OrderOperatorSubComplex>
    IndicesAndCoefficients for    
        Arc<RelativeBoundaryMatrixOracle<DissimilarityMatrix, Filtration, Coefficient, RingOperator, OrderOperatorSubComplex>>
            where
                DissimilarityMatrix: IndicesAndCoefficients<ColIndex=usize, RowIndex=usize, Coefficient=Filtration> + ViewRowAscend + MatrixEntry + Debug + Clone,
                DissimilarityMatrix::EntryMajor: KeyValGet<usize, Filtration> + Debug + Copy,   
                Filtration: Clone + Copy + Debug + PartialOrd + Ord + Hash,
                Coefficient: Clone + Debug,
                RingOperator: Semiring<Coefficient> + Ring<Coefficient> + DivisionRing<Coefficient> + Copy + Debug,  
                OrderOperatorSubComplex: JudgePartialOrder<SimplexFiltered<Filtration>> + SimplexDiameter<Filtration> + Clone
{
    type EntryMajor = (Self::ColIndex, Self::Coefficient);
    type EntryMinor = (Self::RowIndex, Self::Coefficient);    
    type RowIndex = SimplexFiltered<Filtration>; 
    type ColIndex = SimplexFiltered<Filtration>; 
    type Coefficient = Coefficient;
}

/// Implement ViewMajorAscend for `RelativeBoundaryMatrixOracle`
impl <'a, DissimilarityMatrix, Filtration, Coefficient, RingOperator, OrderOperatorSubComplex>
    ViewRowAscend for  
        Arc<RelativeBoundaryMatrixOracle<DissimilarityMatrix, Filtration, Coefficient, RingOperator, OrderOperatorSubComplex>>
            where 
                DissimilarityMatrix: IndicesAndCoefficients<ColIndex=usize, RowIndex=usize, Coefficient=Filtration> + ViewRowAscend + MatrixEntry + Debug + Clone,
                DissimilarityMatrix::EntryMajor: KeyValGet<usize, Filtration> + Debug + Copy,   
                Filtration: Clone + Copy + Debug + PartialOrd + Ord + Hash,
                Coefficient: Clone + Debug,
                RingOperator: Semiring<Coefficient> + Ring<Coefficient> + DivisionRing<Coefficient> + Copy + Debug, 
                OrderOperatorSubComplex: JudgePartialOrder<SimplexFiltered<Filtration>> + SimplexDiameter<Filtration> + Clone
{
    type ViewMajorAscend = LazyOrderedCoboundary<DissimilarityMatrix, Filtration, Coefficient, RingOperator>;
    type ViewMajorAscendIntoIter = Self::ViewMajorAscend; 
    /// 
    /// Get a major (row) view of `RelativeBoundaryMatrixOracle` with column entries in 
    /// strictly ascending order by index. The provided index given by `keymaj` is a 
    /// `SimplexFiltered` struct indexing some row. 
    /// 
    /// This function simply wraps a call to `self.chain_complex_vr_full_space.view_major_ascend()`
    /// 
    fn view_major_ascend(&self, keymaj: Self::RowIndex) -> Self::ViewMajorAscend {
        let arc = Arc::new(self.chain_complex_vr_full_space.clone());
        return arc.view_major_ascend(keymaj);  
    }
}

/// Implement ViewMinorDescend for `RelativeBoundaryMatrixOracle`
impl <'a, DissimilarityMatrix, Filtration, Coefficient, RingOperator, OrderOperatorSubComplex>
    ViewColDescend for  
        Arc<RelativeBoundaryMatrixOracle<DissimilarityMatrix, Filtration, Coefficient, RingOperator, OrderOperatorSubComplex>>
            where 
                DissimilarityMatrix: IndicesAndCoefficients<ColIndex=usize, RowIndex=usize, Coefficient=Filtration> + ViewRowAscend + MatrixEntry + Debug + Clone,
                DissimilarityMatrix::EntryMajor: KeyValGet<usize, Filtration> + Debug + Copy,   
                Filtration: Clone + Copy + Debug + PartialOrd + Ord + Hash,
                Coefficient: Clone + Debug,
                RingOperator: Semiring<Coefficient> + Ring<Coefficient> + DivisionRing<Coefficient> + Copy + Debug, 
                OrderOperatorSubComplex: JudgePartialOrder<SimplexFiltered<Filtration>> + SimplexDiameter<Filtration> + Clone
{
    type ViewMinorDescend = Vec<Self::EntryMinor>;
    type ViewMinorDescendIntoIter = std::vec::IntoIter<Self::EntryMinor>;
    /// 
    /// Get a minor (column) view of `RelativeBoundaryMatrixOracle` with column entries in 
    /// strictly descending order by index. The provided index given by `keymin` is a 
    /// `SimplexFiltered` struct indexing some column. 
    /// 
    /// This function makes a call to `self.chain_complex_vr_full_space.view_minor_descend()`, 
    /// sorts the resulting vector of minor entries by major key using `RelativeBoundaryMatrixRowIndexOrderOperator`
    /// and returns this sorted vector. 
    /// 
    fn view_minor_descend(&self, keymin: Self::ColIndex) -> Self::ViewMinorDescend { 
        let arc = Arc::new(self.chain_complex_vr_full_space.clone());
        let mut minor_view = arc.view_minor_descend(keymin); 
        minor_view.sort_by(
            |lhs, rhs| self.row_index_order_operator.judge_partial_cmp(&rhs.key(), &lhs.key()).unwrap()
        );
        return minor_view; 
    }
} 

/// Implement `VariableSortOracleKeys` for `RelativeBoundaryMatrixOracle`
impl<'a, DissimilarityMatrix, Filtration, Coefficient, RingOperator, OrderOperatorSubComplex>
    VariableSortOracleKeys<
        SimplexFiltered<Filtration>, 
        RelativeBoundaryMatrixRowIndexOrderOperator<Filtration, OrderOperatorFullComplexFiltrationSimplices<DissimilarityMatrix, Filtration>, OrderOperatorSubComplex>, 
        OrderOperatorFullComplexFiltrationSimplices<DissimilarityMatrix, Filtration>, 
        OrderOperatorSubComplex>
        for 
            RelativeBoundaryMatrixOracle<DissimilarityMatrix, Filtration, Coefficient, RingOperator, OrderOperatorSubComplex>
                where 
                    DissimilarityMatrix: IndicesAndCoefficients<ColIndex=usize, RowIndex=usize, Coefficient=Filtration> + ViewRowAscend + MatrixEntry + Debug + Clone,
                    DissimilarityMatrix::EntryMajor: KeyValGet<usize, Filtration> + Debug + Copy,   
                    Filtration: Clone + Copy + Debug + PartialOrd + Ord + Hash,
                    Coefficient: Clone + Debug,
                    RingOperator: Semiring<Coefficient> + Ring<Coefficient> + DivisionRing<Coefficient> + Copy + Debug, 
                    OrderOperatorSubComplex: JudgePartialOrder<SimplexFiltered<Filtration>> + SimplexDiameter<Filtration> + Clone
{ 
    ///
    /// If `major` = `true` (resp. `false`) we return the filtered list of keys 
    /// which index the rows (resp. columns) of `RelativeBoundaryMatrixOracle`.
    /// 
    fn get_sorted_keys_major_or_minor(&self, major: bool) -> Vec<SimplexFiltered<Filtration>> {
        let mut simplices_full_space = self.get_key_list();
        if major {
            simplices_full_space.sort_by(
                |lhs, rhs| self.row_index_order_operator.judge_partial_cmp(lhs,rhs).unwrap()
            ); 
        } else { 
            simplices_full_space.sort_by(
                |lhs, rhs| self.column_index_order_operator.judge_partial_cmp(lhs,rhs).unwrap()
            ); 
        }
        return simplices_full_space; 
    }

    ///
    /// Get all filtered simplices in the full complex from which 
    /// `RelativeBoundaryMatrixOracle` was constructed. 
    /// 
    fn get_key_list(&self) -> Vec<SimplexFiltered<Filtration>> {
        let mut simplices_full_space: Vec<SimplexFiltered<Filtration>> = Vec::new(); 
        for dim in 0..self.max_simplex_dimension+1 { 
            let simplices_fixed_dim: Vec<SimplexFiltered<Filtration>> = self.chain_complex_vr_full_space.cliques_in_lexicographic_order_fixed_dimension(dim as isize).collect_vec(); 
            simplices_full_space.extend(simplices_fixed_dim); 
        }
        return simplices_full_space; 
    }

    ///
    /// Return an instance of `RelativeBoundaryMatrixRowIndexOrderOperator`. 
    /// 
    fn order_operator_key_major_ref(
        &self
    ) -> &RelativeBoundaryMatrixRowIndexOrderOperator<Filtration, OrderOperatorFullComplexFiltrationSimplices<DissimilarityMatrix, Filtration>, OrderOperatorSubComplex> 
    {
        return &self.row_index_order_operator;
    }

    ///
    /// Return an instance of `OrderOperatorFullComplexFiltrationSimplices`. 
    /// 
    fn order_operator_key_minor_ref(&self) -> &OrderOperatorFullComplexFiltrationSimplices<DissimilarityMatrix, Filtration> {
        return &self.column_index_order_operator; 
    }

    ///
    /// Return an instacne of `OrderOperatorSubComplexFiltrationSimplices`. 
    /// 
    fn order_operator_sub_complex_ref(&self) -> &OrderOperatorSubComplex {
        return &self.subspace_filtration_order_operator;
    }
}

/// Implement 'FactorFromArc' for `RelativeBoundaryMatrixOracle`
impl<DissimilarityMatrix, Filtration, Coefficient, RingOperator, OrderOperatorSubComplex>
    FactorFromArc<Self, RingOperator, OrderOperatorFullComplexFiltrationSimplices<DissimilarityMatrix, Filtration>, RelativeBoundaryMatrixRowIndexOrderOperator<Filtration, OrderOperatorFullComplexFiltrationSimplices<DissimilarityMatrix, Filtration>, OrderOperatorSubComplex>>
        for RelativeBoundaryMatrixOracle<DissimilarityMatrix, Filtration, Coefficient, RingOperator, OrderOperatorSubComplex>
            where 
                DissimilarityMatrix: ViewRowAscend + IndicesAndCoefficients<ColIndex=usize, RowIndex=usize, Coefficient=Filtration> + MatrixEntry + Debug + Clone,
                <DissimilarityMatrix as IndicesAndCoefficients>::EntryMajor: KeyValGet<usize, Filtration> + Debug + Clone + Copy, 
                Filtration: Copy + Debug + Ord + Hash,  
                Coefficient: Clone + Debug,
                RingOperator: Semiring<Coefficient> + Ring<Coefficient> + DivisionRing<Coefficient> + Copy + Debug + Clone, 
                OrderOperatorSubComplex: JudgePartialOrder<SimplexFiltered<Filtration>> + SimplexDiameter<Filtration> + Clone
{ 
    /// 
    /// Compute a factorization of the `RelativeBoundaryMatrixOracle` restricted to 
    /// simplices of dimension 0 .. `self.max_simplex_dimension` (inclusive). 
    /// 
    fn factor_from_arc(
        &self, 
    ) -> Umatch<
            Arc<Self>, 
            RingOperator, 
            OrderOperatorByKeyCutsom<
                SimplexFiltered<Filtration>, 
                Coefficient, 
                (SimplexFiltered<Filtration>, Coefficient), 
                OrderOperatorFullComplexFiltrationSimplices<DissimilarityMatrix, Filtration>
            >, 
            OrderOperatorByKeyCutsom<
                SimplexFiltered<Filtration>, 
                Coefficient, 
                (SimplexFiltered<Filtration>, Coefficient), 
                RelativeBoundaryMatrixRowIndexOrderOperator<
                    Filtration, 
                    OrderOperatorFullComplexFiltrationSimplices<DissimilarityMatrix, Filtration>, 
                    OrderOperatorSubComplex
                >
            >
        > 
    {
        let arc = Arc::new(self.clone());
        let umatch = Umatch::factor(
            arc.clone(),
            arc.get_sorted_keys_major_or_minor(true).into_iter().rev(), 
            arc.ring_operator.clone(), 
            arc.row_index_order_operator.clone(), 
            arc.column_index_order_operator.clone()
        );
        return umatch; 
    }
}

//  ===========================================================
//  Struct: RelativeBoundaryMatrixOracleWrapper
//  ===========================================================

#[derive(Clone)]
/// 
/// A wrapper struct for the boundary oracle of a filtered, quotient chain complex. Given the oracle and its 
/// U-Match, this struct provides methods for computing filtered bases of relative cycles, relative boundaries
/// and relative homology. 
/// 
/// Note that the tratis of the generic type `RelativeOracle` allow this structure to take any instance of a matrix 
/// oracle which implements traits identical to those of struct [`RelativeBoundaryMatrixOracle`]. This will allow the 
/// user flexibility and modification when computing PRH. Some examples/cases where this may be helpful: 
/// 
/// - User does not wish to construct type `RelativeOracle` from a dissimilarity matrix. 
/// - User wishes to modify the simplex diamter computation for the full complex or sub complex. 
/// - Use cases which require that the filtration on the subcomplex is not necessarily a restriction 
/// of the filtration on the full complex. 
/// 
/// The standard use case of this struct will be to pass an instance of `RelativeBoundaryMatrixOracle` where the generic 
/// type `OrderOperatorSubComplex` is an instance of [`OrderOperatorSubComplexFiltrationSimplices`]. 
/// 
/// Modified use cases can be contructed in two ways: 
/// 
/// 1. The generic type `RelativeOracle` is an instance of [`RelativeBoundaryMatrixOracle`] whose generic type 
/// `OrderOperatorSubComplex` is a customized instance of [`OrderOperatorSubComplexFiltrationSimplices`] or a custom 
/// order operator. 
/// 2. The generic type `RelativeOracle` is a custom oracle struct implementing the same traits as [`RelativeBoundaryMatrixOracle`]. 
/// 
/// For most use cases, the developers strongly recommend using the first approach. 
/// 
pub struct RelativeBoundaryMatrixOracleWrapper<
    RelativeOracle, Filtration, Coefficient, RingOperator, 
    OrderOperatorOracleKeyMinor, OrderOperatorOracleKeyMajor,   // order operators on keys of `RelativeOracle` ... needed to construct a `row_major::U-Match` struct
    OrderOperatorOracleViewMinor, OrderOperatorOracleViewMajor, // order operators on entries of `RelativeOracle` ... these are constructed when calling `U-Match::factor()`, and are `OrderOperatorAuto` structs which use the underlying order operators on keys of the oralce
    OrderOperatorSubComplex> 
        where 
            // NOTE: these traits bounds are very restrictive in the sense that ANY `RelativeOracle` must have keys of type `SimplexFiltered`
            // Trait bounds required for the RelativeOracle
            // - these traits are needed for U-Match, and ViewMinorDescend / ViewMajorAscend for COMBs
            RelativeOracle: Clone + VariableSortOracleKeys<SimplexFiltered<Filtration>, OrderOperatorOracleKeyMajor, OrderOperatorOracleKeyMinor, OrderOperatorSubComplex> + FactorFromArc<RelativeOracle, RingOperator, OrderOperatorOracleKeyMinor, OrderOperatorOracleKeyMajor>,             
            Arc<RelativeOracle>: ViewRowAscend + ViewColDescend + IndicesAndCoefficients<RowIndex=SimplexFiltered<Filtration>, ColIndex=SimplexFiltered<Filtration>, Coefficient=Coefficient, EntryMajor=(SimplexFiltered<Filtration>, Coefficient), EntryMinor=(SimplexFiltered<Filtration>, Coefficient)>, 
            <Arc<RelativeOracle> as IndicesAndCoefficients>::EntryMajor: Clone + KeyValSet<SimplexFiltered<Filtration>, Coefficient> + KeyValNew<SimplexFiltered<Filtration>, Coefficient> + KeyValGet<SimplexFiltered<Filtration>, Coefficient>, 
            <Arc<RelativeOracle> as IndicesAndCoefficients>::EntryMinor: Clone + KeyValSet<SimplexFiltered<Filtration>, Coefficient> + KeyValNew<SimplexFiltered<Filtration>, Coefficient> + KeyValGet<SimplexFiltered<Filtration>, Coefficient>, 
            // Trait bounds required for the order operators 
            // - the last two need to be specified to access traits of COMBs associated with `self.umatch`
            OrderOperatorOracleKeyMinor: JudgePartialOrder<SimplexFiltered<Filtration>> + JudgePartialOrder<(SimplexFiltered<Filtration>, Coefficient)> + SimplexDiameter<Filtration> + Clone,  // needs to provide a method to compute diameter of a simplex in the FULL complex 
            OrderOperatorSubComplex: JudgePartialOrder<SimplexFiltered<Filtration>> + SimplexDiameter<Filtration> + Clone,                                                                      // needs to provide a method to compute diameter of a simplex in the SUB complex
            OrderOperatorOracleKeyMajor: JudgePartialOrder<SimplexFiltered<Filtration>> + JudgePartialOrder<(SimplexFiltered<Filtration>, Coefficient)> + Clone,                                // needs to be able to compare keys and entries of a boundary matrix oracle and its corresponding COMBs
            OrderOperatorOracleViewMajor: Clone + JudgePartialOrder<(SimplexFiltered<Filtration>, Coefficient)>, 
            OrderOperatorOracleViewMinor: Clone + JudgePartialOrder<(SimplexFiltered<Filtration>, Coefficient)>,
            // Other trait bounds 
            Filtration: Clone + Copy + Debug + PartialOrd + Ord + Hash,
            Coefficient: Clone + Debug + Hash,
            RingOperator: Semiring<Coefficient> + Ring<Coefficient> + DivisionRing<Coefficient> + Copy + Debug
{ 
    /// The underlying `RelativeBoundaryMatrixOracle`.
    pub mapping: RelativeOracle, 
    /// The order operator on the sub complex. 
    pub order_operator_sub_complex: OrderOperatorSubComplex, 
    /// The U-Match Decomposition of `self.mapping`. 
    pub umatch: Umatch<Arc<RelativeOracle>, RingOperator, OrderOperatorOracleViewMajor, OrderOperatorOracleViewMinor>,
    /// Order operator: columns of the target COMB of the U-Match decomposition of a `RelativeBoundaryMatrixOracle`. Contains a reference to the associated U-Match. 
    pub target_comb_order_operator: TargetColumnIndexOrderOperator<RelativeOracle, Filtration, Coefficient, RingOperator, OrderOperatorOracleKeyMinor, 
                                        OrderOperatorOracleKeyMajor, OrderOperatorOracleViewMinor, OrderOperatorOracleViewMajor, OrderOperatorSubComplex>, 
    /// Order operator: columns of the source COMB of the U-Match decomposition of a `RelativeBoundaryMatrixOracle`. Contains a reference to the associated U-Match. 
    pub source_comb_order_operator: SourceColumnIndexOrderOperator<RelativeOracle, Filtration, Coefficient, RingOperator, OrderOperatorOracleKeyMinor, 
                                        OrderOperatorOracleKeyMajor, OrderOperatorOracleViewMinor, OrderOperatorOracleViewMajor, OrderOperatorSubComplex>,   
    
    // NOTE: it is recommended that the hash map order operators are used! 
    /// Order operator: hash map sorter for the columns of the target COMB of the U-Match decomposition of a `RelativeBoundaryMatrixOracle`. Contains owned copy of the sorted vector of indices. 
    pub hash_sorter_for_target_comb_order_operator: HashMapOrderOperator<SimplexFiltered<Filtration>>, 
    /// Order operator: hash map sorter for the columns of the source COMB of the U-Match decomposition of a `RelativeBoundaryMatrixOracle`. Contains owned copy of the sorted vector of indices. 
    pub hash_sorter_for_source_comb_order_operator: HashMapOrderOperator<SimplexFiltered<Filtration>>,
    /// Order operator: hash map sorter for the row indices of `self.mapping`. Contains owned copy of the sorted vector of indices. 
    pub hash_sorter_for_row_indices_of_boundary_oracle: HashMapOrderOperator<SimplexFiltered<Filtration>>, 
    /// Order operator: hash map sorter for the column indices of `self.mapping`. Contains owned copy of the sorted vector of indices. 
    pub hash_sorter_for_column_indices_of_boundary_oracle: HashMapOrderOperator<SimplexFiltered<Filtration>>
}

/// Implementation and methods of `RelativeBoundaryMatrixOracleWrapper`
impl<RelativeOracle, Filtration, Coefficient, RingOperator, OrderOperatorOracleKeyMinor, OrderOperatorOracleKeyMajor, OrderOperatorOracleViewMinor, OrderOperatorOracleViewMajor, OrderOperatorSubComplex> 
    RelativeBoundaryMatrixOracleWrapper<RelativeOracle, Filtration, Coefficient, RingOperator, OrderOperatorOracleKeyMinor, OrderOperatorOracleKeyMajor, OrderOperatorOracleViewMinor, OrderOperatorOracleViewMajor, OrderOperatorSubComplex> 
        where 
            RelativeOracle: Clone + VariableSortOracleKeys<SimplexFiltered<Filtration>, OrderOperatorOracleKeyMajor, OrderOperatorOracleKeyMinor, OrderOperatorSubComplex> + FactorFromArc<RelativeOracle, RingOperator, OrderOperatorOracleKeyMinor, OrderOperatorOracleKeyMajor>,             
            Arc<RelativeOracle>: ViewRowAscend + ViewColDescend + IndicesAndCoefficients<RowIndex=SimplexFiltered<Filtration>, ColIndex=SimplexFiltered<Filtration>, Coefficient=Coefficient, EntryMajor=(SimplexFiltered<Filtration>, Coefficient), EntryMinor=(SimplexFiltered<Filtration>, Coefficient)>, 
            <Arc<RelativeOracle> as IndicesAndCoefficients>::EntryMajor: Clone + KeyValSet<SimplexFiltered<Filtration>, Coefficient> + KeyValNew<SimplexFiltered<Filtration>, Coefficient> + KeyValGet<SimplexFiltered<Filtration>, Coefficient>, 
            <Arc<RelativeOracle> as IndicesAndCoefficients>::EntryMinor: Clone + KeyValSet<SimplexFiltered<Filtration>, Coefficient> + KeyValNew<SimplexFiltered<Filtration>, Coefficient> + KeyValGet<SimplexFiltered<Filtration>, Coefficient>,  
            OrderOperatorOracleKeyMinor: JudgePartialOrder<SimplexFiltered<Filtration>> + JudgePartialOrder<(SimplexFiltered<Filtration>, Coefficient)> + SimplexDiameter<Filtration> + Clone,    
            OrderOperatorSubComplex: JudgePartialOrder<SimplexFiltered<Filtration>> + SimplexDiameter<Filtration> + Clone,       
            OrderOperatorOracleKeyMajor: JudgePartialOrder<SimplexFiltered<Filtration>> + JudgePartialOrder<(SimplexFiltered<Filtration>, Coefficient)> + Clone,                           
            OrderOperatorOracleViewMajor: Clone + JudgePartialOrder<(SimplexFiltered<Filtration>, Coefficient)>, 
            OrderOperatorOracleViewMinor: Clone + JudgePartialOrder<(SimplexFiltered<Filtration>, Coefficient)>,
            Filtration: Clone + Copy + Debug + PartialOrd + Ord + Hash,
            Coefficient: Clone + Debug + Hash,
            RingOperator: Semiring<Coefficient> + Ring<Coefficient> + DivisionRing<Coefficient> + Copy + Debug,
{
    ///
    /// Construct an instance of the `RelativeBoundaryMatrixOracleWrapper`. The 
    /// constructor takes an instance of `RelativeBoundaryMatrixOracle` (or a similar 
    /// matrix oracle) and its Umatch, and packages necessary structs for the computation of 
    /// relative cycles, relative boundaries and relative homology. 
    /// 
    /// Note that all necessary safety checks are or should be performed by the 
    /// constructor for the provided reference to `RelativeOracle`.
    /// 
    pub fn new(
        mapping: RelativeOracle,
        umatch: Umatch<Arc<RelativeOracle>, RingOperator, OrderOperatorOracleViewMajor, OrderOperatorOracleViewMinor>
    ) -> Self 
    
    {  
        // NOTE: the hash map sorters hold references to each sorted list of simplices that one may desire for working with 
        // this moudle. Hence, anytime the user needs a sorted list of simplices, it should be retreived via these structs 
        // rather than lazily generated! 

        // keys of the complex 
        let key_list = mapping.get_key_list();
        // operators for relative feature testing and ordering
        let target_comb_order_operator = TargetColumnIndexOrderOperator::new(
            umatch.clone(), 
            mapping.order_operator_sub_complex_ref().to_owned(), 
            mapping.order_operator_key_minor_ref().to_owned()
        );
        let source_comb_order_operator = SourceColumnIndexOrderOperator::new(
            umatch.clone(), 
            mapping.order_operator_sub_complex_ref().to_owned(), 
            mapping.order_operator_key_minor_ref().to_owned()
        );
        // efficient hash map order operators for repeated sorting via relative feature testing
        let hash_sorter_target = HashMapOrderOperator::new(
            key_list.clone(), 
            target_comb_order_operator.clone()
        );
        let hash_sorter_source = HashMapOrderOperator::new(
            key_list.clone(), 
            source_comb_order_operator.clone()
        );
        // efficient hash map order operators for repeated sorting by row and column index of the boundary oracle
        let key_list = mapping.get_key_list();
        let hash_sorter_row_index = HashMapOrderOperator::new(
            key_list.clone(), 
            mapping.order_operator_key_major_ref().to_owned()
        );
        let hash_sorter_column_index = HashMapOrderOperator::new(
            key_list.clone(), 
            mapping.order_operator_key_minor_ref().to_owned()
        );
        // return oracle wrapper
        RelativeBoundaryMatrixOracleWrapper { 
            mapping: mapping.clone(),
            order_operator_sub_complex: mapping.order_operator_sub_complex_ref().to_owned(), 
            umatch: umatch.clone(), 
            target_comb_order_operator: target_comb_order_operator.clone(), 
            source_comb_order_operator: source_comb_order_operator.clone(),
            hash_sorter_for_target_comb_order_operator: hash_sorter_target.clone(), 
            hash_sorter_for_source_comb_order_operator: hash_sorter_source.clone(),
            hash_sorter_for_row_indices_of_boundary_oracle: hash_sorter_row_index.clone(), 
            hash_sorter_for_column_indices_of_boundary_oracle: hash_sorter_column_index.clone()
        }
    }

    ///
    /// Given `self.mapping` (the boundary matrix of a filtered, quotient chain complex) return the minor keys 
    /// of the source COMB of `self.umatch` (the U-Match factorization of `self.mapping`) which generate the essential
    /// cycles of the filtered simplicial complex given by `self.mapping`. Equivalently, this is the relative homology 
    /// of the simplicial complex at it's maximum diameter. 
    /// 
    /// The keys are identified using the basis matching properties of U-Match. The vector which is returned contains a 
    /// tuple for each generator which gives the key and diameter/filtration of the generator. 
    /// 
    /// NOTE: This function is used primarily for unit testing with relative cycle representatives. 
    /// 
    pub fn essential_cycles(
        &self
    ) -> Vec< (SimplexFiltered<Filtration>, Filtration) >

    { 
        // STEP 1: U-Match factor a `FilteredProductMatrix` to "match" (filtered) bases of relative cycles and relative boundaries.
        // =======================================================================================================================================
        // - The U-Match is given by T'M' = (A^{-1}B)S', where A^{-1}B is the `FilteredProductMatrix` of the inverse target and source 
        // COMBs of `self.umatch`
        // - By match, we mean that we can construct a basis set for relative boundaries that is a subset of a basis set for relative cycles
        // - Interpretation: we can now think of (relative) homology as a difference between basis sets rather than a difference in 
        // span of two vector spaces.
        // - In this case, we do not impose any NEW order on the indices of the inverse target and source COMBs
        // - The index matching, or matching matrix of this U-Match can be used to identify cycles representatives for essential, relative cycles.
        // =======================================================================================================================================
        let matching = self.index_matching_of_factored_comb_product(); 

        // STEP 2: Identify keys of (relative) homological generators 
        // =======================================================================================================================================
        // - If a relative cycle basis has dimension i, then the first i columns of (AT')M' give a basis for them, where AT' is the 
        // matched basis matrix and M' is the matching matrix of `filtered_comb_product_oracle_factored`.
        // - If a relative boundary basis has dimension j, then the first j columns of AT' give a basis for them.
        // - M' is a full rank matching matrix, so the right multiplication (AT')M' acts to permute columns of AT', and possibly scales them 
        // depending on the coefficient ring chosen when constructing the `RelativeOracle`. Thus, AT' and (AT')M' are equivalent up to a 
        // permutation of columns. 
        // NOTE THAT: Calling unwrap on `matching.keymin_to_keymaj()` is always safe. 
        // =======================================================================================================================================
        let matching_minor_keys = self.sort_source_comb_minor_keys_by_relative_cycle_birth();  
        let source_order_operator = self.source_comb_order_operator_ref().to_owned();
        let target_order_operator = self.target_comb_order_operator_ref().to_owned();
        let mut relative_homology_basis: Vec< (SimplexFiltered<Filtration>, Filtration) > = Vec::new(); 
        let mut cutoff_key_found: bool = false; 
        for key in matching_minor_keys { 
            // only proceed if `key` gives a relative cycle in columns of AT'M' (and thus in the columns of the source COMB of `self.umatch`)
            // given the total order on `matching_minor_keys`, then this loop eventually does nothing except check the boolean. 
            if !cutoff_key_found { 
                if !source_order_operator.is_relative_cycle(&key) { 
                    cutoff_key_found = true; 
                }
                else { 
                    // get major key matched to this minor key 
                    // check to see if it is NOT a relative boundary in the columns of AT (and thus in the columns of the target COMB of `self.umatch`). 
                    let row_index_of_nonzero_entry = matching.keymin_to_keymaj(&key).unwrap(); 
                    if !target_order_operator.is_relative_boundary(&row_index_of_nonzero_entry) { 
                        // =======================================================================================================================================
                        // - If we made it to this if statement, then the following three facts are all true, given the index matching 
                        // (r,c) = (row_index_of_nonzero_entry, key): 
                        //      1) COL_{r}(AT') = COL_{c}(AT'M') since M' a permutation matrix, 
                        //      2) COL_{r}(AT') is not a (relative) boundary, 
                        //      3) COL_{c}(AT'M') is a (relative) cycle. 
                        // Thus, we have found the key of a (relative) homological generator! 
                        // =======================================================================================================================================
                        // STEP 3: Extract cycle representatives 
                        // ======================================================================================================================================= 
                        // - We recover the required columns of AT' without permuting via the multiplcation (AT')M'. We use the fact that AT'M'[:,c] = AT'[:,r]
                        // if and only if M'[:,c] has its nonzero coefficient at row_index = r. 
                        // - There are three ways to construct the cycle representative: 
                        // 1) multiply to get AT'[:,r] ... This is a linear combination of the rows of A scaled by the column vector T'[:,r]
                        // 2) multiply to get BS'[:,c] ... works since we have T'M' = (A^{-1}B)S' ==> AT'M' = BS'. This is a linear combination of the rows of B scaled by 
                        // the column vector S'[:c]
                        // 3) get column vector with minor index `key` from the source comb of original U-Match ... works by inductive structure of the matched basis!  
                        // NOTE: here, we just return the key, rather than the actual generator! 
                        // =======================================================================================================================================
                        relative_homology_basis.push( (key.clone(), source_order_operator.relative_cycle_birth(&key).unwrap()) ); 
                    }
                }
            }
        }
        return relative_homology_basis; 
    }

    ///
    /// Return the minor keys of the source COMB of the U-Match factorization of 
    /// self (the relative boundary matrix oracle) where: 
    /// 
    /// (i.) keys are sorted by increasing birth of their corresponding column 
    /// views as relative cycles.
    /// 
    /// (ii.) keys which correspond to columns that are never born as relative cycles 
    /// are removed.
    /// 
    /// The resulting list of minor keys may be used to obtain a filtered relative 
    /// cycle basis from the columns of the source COMB via the `ViewsMinorDescend`
    /// trait of the `CombDomain` struct. 
    ///  
    pub fn filtered_relative_cycle_basis(&self) -> Vec<SimplexFiltered<Filtration>> {
        let minor_keys_filtered = self.sort_source_comb_minor_keys_by_relative_cycle_birth(); 
        let return_vec = minor_keys_filtered.into_iter().filter_map(|x: SimplexFiltered<Filtration>| {
            if self.source_comb_order_operator.relative_cycle_birth(&x).is_some() { 
                Some(x)
            } else { 
                None
            }
        }).collect_vec(); 
        return return_vec; 
    }

    ///
    /// Return the minor keys of the target COMB of the U-Match factorization of 
    /// self (the relative boundary matrix oracle) where: 
    /// 
    /// (i.) keys are sorted by increasing birth of their corresponding column 
    /// views as relative boundaries.
    /// 
    /// (ii.) keys which correspond to columns that are never born as relative boundaries 
    /// are removed.
    /// 
    /// The resulting list of minor keys may be used to obtain a filtered relative 
    /// boundary basis from the columns of the target COMB via the `ViewsMinorDescend`
    /// trait of the `CombCodomain` struct. 
    ///  
    pub fn filtered_relative_boundary_basis(&self) -> Vec<SimplexFiltered<Filtration>> { 
        let minor_keys_filtered = self.sort_target_comb_minor_keys_by_relative_boundary_birth(); 
        let return_vec = minor_keys_filtered.into_iter().filter_map(|x: SimplexFiltered<Filtration>| {
            if self.target_comb_order_operator.relative_boundary_birth(&x).is_some() { 
                Some(x)
            } else { 
                None
            }
        }).collect_vec(); 
        return return_vec;  
    }

    // TODO: need to rewrite/refactor the basis matching step!!

    // ====================
    // lazy basis matching
    // ====================

    ///
    /// Return the lazy product T^{-1}S where T and S are, respectively, the target and source COMBs for a U-match TM = DS where 
    /// D is the boundary matrix of a (filtered) quotient chain complex. 
    /// 
    fn lazy_comb_product(
        &self
    ) -> BimajorProductMatrix<
            CombCodomainInv<'_, Arc<RelativeOracle>, RingOperator, OrderOperatorOracleViewMajor, OrderOperatorOracleViewMinor>, 
            CombDomain<'_, Arc<RelativeOracle>, RingOperator, OrderOperatorOracleViewMajor, OrderOperatorOracleViewMinor>, 
            RingOperator,
            OrderOperatorOracleKeyMajor, 
            OrderOperatorOracleKeyMinor
        >

    { 
        BimajorProductMatrix::new(
            self.umatch.comb_codomain_inv(), 
            self.umatch.comb_domain(), 
            self.umatch.ring_operator(), 
            self.mapping.get_sorted_keys_major_or_minor(true),                     
            self.mapping.order_operator_key_major_ref().to_owned(),    
            self.mapping.order_operator_key_minor_ref().to_owned()  
        )
    }

    ///
    /// Return the U-Match of a lazy `FilteredProductMatrix` where, given the U-Match of the boundary matrix of a filtered, 
    /// quotient chain complex: 
    /// 
    /// - `Left` is the inverse Target COMB
    /// - `Right` is the source COMB 
    /// 
    pub fn lazy_comb_product_factored(
        &self
    ) -> Umatch<
            Arc<BimajorProductMatrix<
                CombCodomainInv<'_, Arc<RelativeOracle>, RingOperator, OrderOperatorOracleViewMajor, OrderOperatorOracleViewMinor>, 
                CombDomain<'_, Arc<RelativeOracle>, RingOperator, OrderOperatorOracleViewMajor, OrderOperatorOracleViewMinor>, 
                RingOperator, 
                OrderOperatorOracleKeyMajor, 
                OrderOperatorOracleKeyMinor
            >>, 
            RingOperator, 
            OrderOperatorByKeyCutsom<SimplexFiltered<Filtration>, Coefficient, (SimplexFiltered<Filtration>, Coefficient), OrderOperatorOracleKeyMinor>, 
            OrderOperatorByKeyCutsom<SimplexFiltered<Filtration>, Coefficient, (SimplexFiltered<Filtration>, Coefficient), OrderOperatorOracleKeyMajor>
        >
        
    { 
        let comb_product = Arc::new(self.lazy_comb_product()); 
        Umatch::factor(
            comb_product.clone(),                                                          
            comb_product.sorted_major_keys_of_product.clone().into_iter().rev(), 
            comb_product.ring_operator.clone(), 
            comb_product.order_operator_key_major.clone(),
            comb_product.order_operator_view_major.clone() 
        )
    }

    ///
    /// Construct the matched basis associated with the boundary matrix of a filtered, quotient chain complex. This is 
    /// a matrix whose columns contain a basis for all relative cycles and relative boundary of the associated filtered
    /// quotient space. 
    /// 
    /// The user must provide a `comb_product_factored` struct, which is the U-Match of a lazy `FilteredProductMatrix` oracle 
    /// as returned by the function `RelativeBoundaryMatrixOracleWrapper::comb_product_factored()`.
    /// 
    pub fn lazy_matched_basis<'a,'b:'a>(
        &'a self, 
        comb_product_factored: &'b Umatch<
            Arc<BimajorProductMatrix<
                CombCodomainInv<'a, Arc<RelativeOracle>, RingOperator, OrderOperatorOracleViewMajor, OrderOperatorOracleViewMinor>, 
                CombDomain<'a, Arc<RelativeOracle>, RingOperator, OrderOperatorOracleViewMajor, OrderOperatorOracleViewMinor>, 
                RingOperator, 
                OrderOperatorOracleKeyMajor, 
                OrderOperatorOracleKeyMinor
            >>, 
            RingOperator, 
            OrderOperatorByKeyCutsom<SimplexFiltered<Filtration>, Coefficient, (SimplexFiltered<Filtration>, Coefficient), OrderOperatorOracleKeyMinor>, 
            OrderOperatorByKeyCutsom<SimplexFiltered<Filtration>, Coefficient, (SimplexFiltered<Filtration>, Coefficient), OrderOperatorOracleKeyMajor>
        >
    ) -> 
    
    ( 
        // the matched basis
        BimajorProductMatrix<
        CombCodomain<'a, Arc<RelativeOracle>, RingOperator, OrderOperatorOracleViewMajor, OrderOperatorOracleViewMinor>, 
        CombCodomain<'a, 
            Arc<BimajorProductMatrix<
                CombCodomainInv<'a, Arc<RelativeOracle>, RingOperator, OrderOperatorOracleViewMajor, OrderOperatorOracleViewMinor>, 
                CombDomain<'a, Arc<RelativeOracle>, RingOperator, OrderOperatorOracleViewMajor, OrderOperatorOracleViewMinor>, 
                RingOperator, 
                OrderOperatorOracleKeyMajor, 
                OrderOperatorOracleKeyMinor
            >>,
            RingOperator, 
            OrderOperatorByKeyCutsom<SimplexFiltered<Filtration>, Coefficient, (SimplexFiltered<Filtration>, Coefficient), OrderOperatorOracleKeyMinor>, 
            OrderOperatorByKeyCutsom<SimplexFiltered<Filtration>, Coefficient, (SimplexFiltered<Filtration>, Coefficient), OrderOperatorOracleKeyMajor>
        >, 
        RingOperator, 
        OrderOperatorOracleKeyMajor, 
        OrderOperatorOracleKeyMinor
        >, 
        // the target COMB of the basis matching U-match --> used for the change of basis from PH to PRH
        CombCodomain<'a, 
            Arc<BimajorProductMatrix<
                CombCodomainInv<'a, Arc<RelativeOracle>, RingOperator, OrderOperatorOracleViewMajor, OrderOperatorOracleViewMinor>, 
                CombDomain<'a, Arc<RelativeOracle>, RingOperator, OrderOperatorOracleViewMajor, OrderOperatorOracleViewMinor>, 
                RingOperator, 
                OrderOperatorOracleKeyMajor, 
                OrderOperatorOracleKeyMinor
            >>, 
            RingOperator, 
            OrderOperatorByKeyCutsom<SimplexFiltered<Filtration>, Coefficient, (SimplexFiltered<Filtration>, Coefficient), OrderOperatorOracleKeyMinor>, 
            OrderOperatorByKeyCutsom<SimplexFiltered<Filtration>, Coefficient, (SimplexFiltered<Filtration>, Coefficient), OrderOperatorOracleKeyMajor>
        > 
    )

    {
        let target_comb_a = self.umatch.comb_codomain(); 
        let target_comb_t = comb_product_factored.comb_codomain(); 
        let matched_basis_matrix = BimajorProductMatrix::new(
            target_comb_a,   
            target_comb_t.clone(), 
            self.umatch.ring_operator(), 
            self.hash_sorter_for_row_indices_of_boundary_oracle.sorted.clone(),              
            self.mapping.order_operator_key_major_ref().to_owned(), 
            self.mapping.order_operator_key_minor_ref().to_owned()
        ); 
        return (matched_basis_matrix, target_comb_t); 
    }

    // END: lazy basis matching ===================

    // ======================
    // sparse basis matching
    // - NOTE for developers: this code (following three functions) is not yet safe for python side usage!
    // ======================

    ///
    /// Return the CSR product A^{-1}B where, given the U-match of the boundary matrix of a filtered, quotient chain complex, A and B respectively are: 
    /// 
    /// - the inverse target COMB with major keys sorted using the associated instance of [`TargetColumnIndexOrderOperator`]
    /// - the source COMB with minor keys sorted using the associated instance of [`SourceColumnIndexOrderOperator`]
    /// 
    fn sparse_comb_product(
        &self
    ) -> BimajorSparseProductMatrix<
            CombCodomainInv<'_, Arc<RelativeOracle>, RingOperator, OrderOperatorOracleViewMajor, OrderOperatorOracleViewMinor>, 
            CombDomain<'_, Arc<RelativeOracle>, RingOperator, OrderOperatorOracleViewMajor, OrderOperatorOracleViewMinor>, 
            RingOperator, 
            HashMapOrderOperator<SimplexFiltered<Filtration>>, 
            HashMapOrderOperator<SimplexFiltered<Filtration>>
        >

    {  
        BimajorSparseProductMatrix::new(
            self.umatch.comb_codomain_inv(), 
            self.umatch.comb_domain(), 
            self.umatch.ring_operator(), 
            self.hash_sorter_for_target_comb_order_operator.sorted.clone(), 
            self.hash_sorter_for_source_comb_order_operator.sorted.clone(),
            self.hash_sorter_for_target_comb_order_operator.to_owned(), 
            self.hash_sorter_for_source_comb_order_operator.to_owned(), 
        )
    }

    ///
    /// Return the U-Match of a sparse, explicitly sorted, product oracle where, given the U-Match of the boundary matrix of a filtered, 
    /// quotient chain complex: 
    /// 
    /// - `Left` is the inverse Target COMB with row indices sorted using the associated instance of [`TargetColumnIndexOrderOperator`]
    /// - `Right` is the source COMB with column indices sorted using the associated instance of [`SourceColumnIndexOrderOperator`]
    /// 
    pub fn sparse_comb_product_factored(
        &self
    ) -> Umatch<
            Arc<BimajorSparseProductMatrix<
                CombCodomainInv<'_, Arc<RelativeOracle>, RingOperator, OrderOperatorOracleViewMajor, OrderOperatorOracleViewMinor>, 
                CombDomain<'_, Arc<RelativeOracle>, RingOperator, OrderOperatorOracleViewMajor, OrderOperatorOracleViewMinor>, 
                RingOperator, 
                HashMapOrderOperator<SimplexFiltered<Filtration>>, 
                HashMapOrderOperator<SimplexFiltered<Filtration>>
            >>, 
            RingOperator, 
            OrderOperatorByKeyCutsom<SimplexFiltered<Filtration>, Coefficient, (SimplexFiltered<Filtration>, Coefficient), HashMapOrderOperator<SimplexFiltered<Filtration>>>, 
            OrderOperatorByKeyCutsom<SimplexFiltered<Filtration>, Coefficient, (SimplexFiltered<Filtration>, Coefficient), HashMapOrderOperator<SimplexFiltered<Filtration>>>
        > 
    { 
        let comb_product = Arc::new(self.sparse_comb_product()); 
        Umatch::factor(
            comb_product.clone(),                                                          
            comb_product.sorted_row_indices.clone().into_iter().rev(), 
            self.umatch.ring_operator(), 
            self.hash_sorter_for_target_comb_order_operator.clone(), // self.target_comb_order_operator.to_owned(), 
            self.hash_sorter_for_source_comb_order_operator.clone() // self.source_comb_order_operator.to_owned(),  
        )
    }

    ///
    /// Construct the matched basis associated with the boundary matrix of a filtered, quotient chain complex. This is 
    /// a matrix whose columns contain a basis for all relative cycles and relative boundary of the associated filtered
    /// quotient space. 
    /// 
    /// The user must provide a `comb_product_factored` struct, which is the U-Match of a lazy `FilteredProductMatrix` oracle 
    /// as returned by the function `RelativeBoundaryMatrixOracleWrapper::comb_product_factored()`.
    /// 
    pub fn sparse_matched_basis<'a,'b:'a>(
        &'a self, 
        comb_product_factored: &'b Umatch<
            Arc<BimajorSparseProductMatrix<
                CombCodomainInv<'a, Arc<RelativeOracle>, RingOperator, OrderOperatorOracleViewMajor, OrderOperatorOracleViewMinor>, 
                CombDomain<'a, Arc<RelativeOracle>, RingOperator, OrderOperatorOracleViewMajor, OrderOperatorOracleViewMinor>, 
                RingOperator, 
                HashMapOrderOperator<SimplexFiltered<Filtration>>, 
                HashMapOrderOperator<SimplexFiltered<Filtration>>
            >>, 
            RingOperator, 
            OrderOperatorByKeyCutsom<SimplexFiltered<Filtration>, Coefficient, (SimplexFiltered<Filtration>, Coefficient), HashMapOrderOperator<SimplexFiltered<Filtration>>>, 
            OrderOperatorByKeyCutsom<SimplexFiltered<Filtration>, Coefficient, (SimplexFiltered<Filtration>, Coefficient), HashMapOrderOperator<SimplexFiltered<Filtration>>>
        > 
    ) -> 
    
    ( 
        // the matched basis
        BimajorProductMatrix<
            CombCodomain<'a, Arc<RelativeOracle>, RingOperator, OrderOperatorOracleViewMajor, OrderOperatorOracleViewMinor>, 
            CombCodomain<'a, 
                Arc<BimajorSparseProductMatrix<
                    CombCodomainInv<'a, Arc<RelativeOracle>, RingOperator, OrderOperatorOracleViewMajor, OrderOperatorOracleViewMinor>, 
                    CombDomain<'a, Arc<RelativeOracle>, RingOperator, OrderOperatorOracleViewMajor, OrderOperatorOracleViewMinor>, 
                    RingOperator, 
                    HashMapOrderOperator<SimplexFiltered<Filtration>>, 
                    HashMapOrderOperator<SimplexFiltered<Filtration>>
                >>, 
                RingOperator, 
                OrderOperatorByKeyCutsom<SimplexFiltered<Filtration>, Coefficient, (SimplexFiltered<Filtration>, Coefficient), HashMapOrderOperator<SimplexFiltered<Filtration>>>, 
                OrderOperatorByKeyCutsom<SimplexFiltered<Filtration>, Coefficient, (SimplexFiltered<Filtration>, Coefficient), HashMapOrderOperator<SimplexFiltered<Filtration>>>
            >, 
            RingOperator, 
            OrderOperatorOracleKeyMajor,
            HashMapOrderOperator<SimplexFiltered<Filtration>> 
        >, 
        // the target COMB of the basis matching U-match --> used for the change of basis from PH to PRH
        CombCodomain<'a, 
            Arc<BimajorSparseProductMatrix<
                CombCodomainInv<'a, Arc<RelativeOracle>, RingOperator, OrderOperatorOracleViewMajor, OrderOperatorOracleViewMinor>, 
                CombDomain<'a, Arc<RelativeOracle>, RingOperator, OrderOperatorOracleViewMajor, OrderOperatorOracleViewMinor>, 
                RingOperator, 
                HashMapOrderOperator<SimplexFiltered<Filtration>>, 
                HashMapOrderOperator<SimplexFiltered<Filtration>>
            >>, 
            RingOperator, 
            OrderOperatorByKeyCutsom<SimplexFiltered<Filtration>, Coefficient, (SimplexFiltered<Filtration>, Coefficient), HashMapOrderOperator<SimplexFiltered<Filtration>>>, 
            OrderOperatorByKeyCutsom<SimplexFiltered<Filtration>, Coefficient, (SimplexFiltered<Filtration>, Coefficient), HashMapOrderOperator<SimplexFiltered<Filtration>>>
        >
    )

    {
        let target_comb_a = self.umatch.comb_codomain(); 
        let target_comb_t = comb_product_factored.comb_codomain(); 
        let matched_basis_matrix = BimajorProductMatrix::new(
            target_comb_a,   
            target_comb_t.clone(), 
            self.umatch.ring_operator(), 
            self.hash_sorter_for_row_indices_of_boundary_oracle.sorted.clone(), 
            self.mapping.order_operator_key_major_ref().to_owned(), 
            self.hash_sorter_for_source_comb_order_operator.clone() // --> this is to ensure that inner products of rows and columns are correct! 
        ); 
        return (matched_basis_matrix, target_comb_t); 
    }

    // END: sparse basis matching ===================

    /// 
    /// Return the minor keys of the target COMB of the U-Match factorization of 
    /// a `RelativeBoundaryMatrixOracle` where keys are sorted by increasing 
    /// birth of their corresponding column views as relative boundaries.
    /// 
    pub fn sort_target_comb_minor_keys_by_relative_boundary_birth(&self) -> Vec<SimplexFiltered<Filtration>> { 
        return self.hash_sorter_for_target_comb_order_operator.sorted.clone();
    }

    /// 
    /// Return the minor keys of the source COMB of the U-Match factorization of 
    /// a `RelativeBoundaryMatrixOracle` where keys are sorted by increasing 
    /// birth of their corresponding column views as relative cycles.
    /// 
    pub fn sort_source_comb_minor_keys_by_relative_cycle_birth(&self) -> Vec<SimplexFiltered<Filtration>> { 
        return self.hash_sorter_for_source_comb_order_operator.sorted.clone();
    }

    ///
    /// Given a chain which generates a relative homology class, it is most intuitive for the chain to 
    /// not include any simplices contained in the subcomplex. However, it is not always the case that 
    /// generators obtained via basis matching meet this criteria. In the case that this happens, this 
    /// function may be used to provide an alternative cycle representative which excludes subcomplex 
    /// simplices.   
    /// 
    pub fn trim_relative_chain(
        &self, 
        relative_chain: Vec< (SimplexFiltered<Filtration>, Coefficient) >
    ) -> Vec< (SimplexFiltered<Filtration>, Coefficient) >

    { 
        let mut trimmed_cycle: Vec< (SimplexFiltered<Filtration>, Coefficient) > = Vec::new(); 
        let mut relative_chain_iter: std::vec::IntoIter<(SimplexFiltered<Filtration>, Coefficient)> = relative_chain.into_iter();
        for _ in 0..relative_chain_iter.clone().len() { 
            // we only call unwrap for the length of the vector, so it is always safe! 
            let curr_entry: (SimplexFiltered<Filtration>, Coefficient) = relative_chain_iter.next().unwrap(); 
            // if simplex of the current entry does not have a sub complex diameter, add it to the generator to return
            if !self.order_operator_sub_complex.diameter(curr_entry.0.vertices()).is_some() { 
                trimmed_cycle.push(curr_entry); 
            }
        }
        return trimmed_cycle; 
    }

    ///
    /// Retrun a reference to the U-Match factorization of the wrapped `RelativeBoundaryMatrixOracle`.
    /// 
    pub fn factored_ref(
        &self
    ) -> &Umatch<Arc<RelativeOracle>, RingOperator, OrderOperatorOracleViewMajor, OrderOperatorOracleViewMinor>
    { 
        return &self.umatch; 
    }

    ///
    /// Return a reference to the target COMB order operator.
    /// 
    pub fn target_comb_order_operator_ref(
        &self
    ) -> &TargetColumnIndexOrderOperator<RelativeOracle, Filtration, Coefficient, RingOperator, OrderOperatorOracleKeyMinor, 
            OrderOperatorOracleKeyMajor, OrderOperatorOracleViewMinor, OrderOperatorOracleViewMajor, OrderOperatorSubComplex>
    
    { 
        return &self.target_comb_order_operator;
    }

    ///
    /// Return a reference to the source COMB order operator.
    ///
    pub fn source_comb_order_operator_ref(
        &self
    ) -> &SourceColumnIndexOrderOperator<RelativeOracle, Filtration, Coefficient, RingOperator, OrderOperatorOracleKeyMinor, 
            OrderOperatorOracleKeyMajor, OrderOperatorOracleViewMinor, OrderOperatorOracleViewMajor, OrderOperatorSubComplex>
    
    { 
        return &self.source_comb_order_operator; 
    }

    ///
    /// Return a reference to the simplices of the complex sorted by the order operator `Self::OrderOperatorOracleViewMajor`.
    ///
    pub fn simplices_full_complex_order_ref(
        &self
    ) -> &Vec<SimplexFiltered<Filtration>>
    
    { 
        return &self.hash_sorter_for_column_indices_of_boundary_oracle.sorted; 
    }

    ///
    /// Return a reference to the simplices of the complex sorted by the order operator [`OrderOperatorOracleViewMinor`].
    ///
    pub fn simplices_subcomplex_order_ref(
        &self
    ) -> &Vec<SimplexFiltered<Filtration>>
    
    { 
        return &self.hash_sorter_for_row_indices_of_boundary_oracle.sorted; 
    }

    ///
    /// Return a reference to the simplices of the complex sorted by the order operator [`SourceColumnIndexOrderOperator`].
    ///
    pub fn simplices_relative_cycle_order_ref(
        &self
    ) -> &Vec<SimplexFiltered<Filtration>>
    
    { 
        return &self.hash_sorter_for_source_comb_order_operator.sorted; 
    }

    ///
    /// Return a reference to the simplices of the complex sorted by the order operator [`TargetColumnIndexOrderOperator`].
    ///
    pub fn simplices_relative_boundary_order_ref(
        &self
    ) -> &Vec<SimplexFiltered<Filtration>>
    
    { 
        return &self.hash_sorter_for_target_comb_order_operator.sorted; 
    }

    ///
    /// Return the index matching, or matching matrix, of the U-Match of a lazy `FilteredProductMatrix` where, given 
    /// the U-Match of the boundary matrix of a filtered, quotient chain complex: 
    /// 
    /// - `Left` is the inverse Target COMB
    /// - `Right` is the source COMB 
    /// 
    /// NOTE: this function should be used with caution, as each call computes a new U-match object. 
    /// 
    pub fn index_matching_of_factored_comb_product(
        &self
    ) -> GeneralizedMatchingArrayWithMajorOrdinals<SimplexFiltered<Filtration>, SimplexFiltered<Filtration>, Coefficient>
    
    { 
        self.lazy_comb_product_factored().matching_ref().to_owned()
    }
}

//  ===========================================================
//  Trait Implementations for `BimajorProductMatrix`
//  ===========================================================

// - allows product oracle to be U-Match factored from an Arc reference
// - note that views of the product matrix are wrapped to vectors of (Index, Coefficient) pairs
// - views are formatted in this way as simplified linear combinations of COMB views have lifetime parameters attached to them
// - thus, wrapping these views as vectors make this module much cleaner to read and write!

/// Implement `IndicesAndCoefficients` for `BimajorProductMatrix`
impl<'a, LeftMatrix, RightMatrix, RingOperator, OrderOperatorKeyMajor, OrderOperatorViewMajor>
    IndicesAndCoefficients for 
        Arc<BimajorProductMatrix<LeftMatrix, RightMatrix, RingOperator, OrderOperatorKeyMajor, OrderOperatorViewMajor>>
            where 
                LeftMatrix: ViewRowAscend + IndicesAndCoefficients,  
                LeftMatrix::ViewMajorAscend: IntoIterator,
                LeftMatrix::EntryMajor: KeyValGet<LeftMatrix::ColIndex, LeftMatrix::Coefficient>,
                RightMatrix: ViewRowAscend + IndicesAndCoefficients<Coefficient = LeftMatrix::Coefficient, RowIndex = LeftMatrix::ColIndex>,  
                RightMatrix::ViewMajorAscend: IntoIterator, 
                RightMatrix::EntryMajor: KeyValGet<RightMatrix::ColIndex, RightMatrix::Coefficient>,
                RightMatrix::ColIndex: Clone, 
                RightMatrix::Coefficient: Clone,
                RingOperator: Clone + Semiring<LeftMatrix::Coefficient>,
                OrderOperatorViewMajor: JudgePartialOrder<RightMatrix::EntryMajor> + Clone, 
                OrderOperatorKeyMajor: JudgePartialOrder<LeftMatrix::RowIndex> + JudgePartialOrder<LeftMatrix::EntryMinor> + Clone
{ 
    type EntryMajor = RightMatrix::EntryMajor;
    type EntryMinor = LeftMatrix::EntryMinor;    
    type RowIndex = LeftMatrix::RowIndex;
    type ColIndex = RightMatrix::ColIndex; 
    type Coefficient = LeftMatrix::Coefficient;
}

/// Implement `ViewRowAscend` for `BimajorProductMatrix`
impl<'a, LeftMatrix, RightMatrix, RingOperator, OrderOperatorKeyMajor, OrderOperatorViewMajor>
    ViewRowAscend for 
        Arc<BimajorProductMatrix<LeftMatrix, RightMatrix, RingOperator, OrderOperatorKeyMajor, OrderOperatorViewMajor>>
            where 
                LeftMatrix: ViewRowAscend + IndicesAndCoefficients,  
                LeftMatrix::ViewMajorAscend: IntoIterator,
                LeftMatrix::EntryMajor: KeyValGet<LeftMatrix::ColIndex, LeftMatrix::Coefficient>,
                RightMatrix: ViewRowAscend + IndicesAndCoefficients<Coefficient = LeftMatrix::Coefficient, RowIndex = LeftMatrix::ColIndex>,  
                RightMatrix::ViewMajorAscend: IntoIterator, 
                RightMatrix::EntryMajor: KeyValGet<RightMatrix::ColIndex, RightMatrix::Coefficient> + KeyValSet<RightMatrix::ColIndex, RightMatrix::Coefficient>,
                RightMatrix::ColIndex: Clone + PartialEq, 
                RightMatrix::Coefficient: Clone,
                RingOperator: Clone + Semiring<LeftMatrix::Coefficient>,
                OrderOperatorViewMajor: JudgePartialOrder<RightMatrix::EntryMajor> + Clone, 
                OrderOperatorKeyMajor: JudgePartialOrder<LeftMatrix::RowIndex> + JudgePartialOrder<LeftMatrix::EntryMinor> + Clone
{
    type ViewMajorAscend = Vec<Self::EntryMajor>;
    type ViewMajorAscendIntoIter = std::vec::IntoIter<Self::EntryMajor>;
    ///
    /// Obtain a major view of the `BimajorProductMatirx`.
    /// 
    fn view_major_ascend(&self, index: Self::RowIndex) -> Self::ViewMajorAscend {
        BimajorProductMatrix::view_major_ascend(&self, index).collect_vec()
    }
}

/// Implement `ViewColDescend` for `BimajorProductMatrix` 
impl<'a, LeftMatrix, RightMatrix, RingOperator, OrderOperatorKeyMajor, OrderOperatorViewMajor>
    ViewColDescend for 
        Arc<BimajorProductMatrix<LeftMatrix, RightMatrix, RingOperator, OrderOperatorKeyMajor, OrderOperatorViewMajor>>
            where 
                LeftMatrix: ViewColDescend + ViewRowAscend + IndicesAndCoefficients,  
                LeftMatrix::ViewMajorAscend: IntoIterator,
                LeftMatrix::RowIndex: Clone + PartialEq,
                LeftMatrix::EntryMajor: KeyValGet<LeftMatrix::ColIndex, LeftMatrix::Coefficient>,
                LeftMatrix::EntryMinor: KeyValGet<LeftMatrix::RowIndex, LeftMatrix::Coefficient> + KeyValSet<LeftMatrix::RowIndex, LeftMatrix::Coefficient>, 
                RightMatrix: ViewColDescend + ViewRowAscend + IndicesAndCoefficients<Coefficient = LeftMatrix::Coefficient, RowIndex = LeftMatrix::ColIndex>,  
                RightMatrix::ViewMajorAscend: IntoIterator, 
                RightMatrix::EntryMajor: KeyValGet<RightMatrix::ColIndex, RightMatrix::Coefficient> + KeyValSet<RightMatrix::ColIndex, RightMatrix::Coefficient>,
                RightMatrix::EntryMinor: KeyValGet<RightMatrix::RowIndex, RightMatrix::Coefficient>,
                RightMatrix::ColIndex: Clone + PartialEq, 
                RightMatrix::Coefficient: Clone,
                RingOperator: Clone + Semiring<LeftMatrix::Coefficient>,
                OrderOperatorViewMajor: JudgePartialOrder<RightMatrix::EntryMajor> + Clone, 
                OrderOperatorKeyMajor: JudgePartialOrder<LeftMatrix::RowIndex> + JudgePartialOrder<LeftMatrix::EntryMinor> + Clone
{
    type ViewMinorDescend = Vec<Self::EntryMinor>;
    type ViewMinorDescendIntoIter = std::vec::IntoIter<Self::EntryMinor>;
    ///
    /// Obtain a minor view of the `BimajorProductMatrix`. 
    /// 
    fn view_minor_descend(&self, index: Self::ColIndex) -> Self::ViewMinorDescend {  
        BimajorProductMatrix::view_minor_descend(&self, index).collect_vec()
    }
} 

//  ===========================================================
//  Trait Implementations for `BimajorProductMatrix`
//  ===========================================================

// - allows product oracle to be U-Match factored from an Arc reference
// - note that views of the product matrix are wrapped to vectors of (Index, Coefficient) pairs
// - views are formatted in this way as simplified linear combinations of COMB views have lifetime parameters attached to them
// - thus, wrapping these views as vectors make this module much cleaner to read and write!

/// Implement `IndicesAndCoefficients` for `BimajorSparseProductMatrix`
impl<'a, LeftMatrix, RightMatrix, RingOperator, OrderOperatorKeyMajor, OrderOperatorViewMajor>
    IndicesAndCoefficients for 
        Arc<BimajorSparseProductMatrix<LeftMatrix, RightMatrix, RingOperator, OrderOperatorKeyMajor, OrderOperatorViewMajor>>
            where 
                LeftMatrix: ViewColDescend + ViewRowAscend + IndicesAndCoefficients,  
                LeftMatrix::ViewMajorAscend: IntoIterator,
                LeftMatrix::EntryMajor: KeyValGet<LeftMatrix::ColIndex, LeftMatrix::Coefficient>,
                LeftMatrix::EntryMinor: KeyValGet<LeftMatrix::RowIndex, LeftMatrix::Coefficient> + KeyValSet<LeftMatrix::RowIndex, LeftMatrix::Coefficient>,
                LeftMatrix::RowIndex: Eq + Hash + Clone,
                RightMatrix: ViewColDescend + ViewRowAscend + IndicesAndCoefficients<Coefficient = LeftMatrix::Coefficient, RowIndex = LeftMatrix::ColIndex>,  
                RightMatrix::ViewMajorAscend: IntoIterator, 
                RightMatrix::EntryMajor: KeyValGet<RightMatrix::ColIndex, RightMatrix::Coefficient> + KeyValSet<RightMatrix::ColIndex, RightMatrix::Coefficient>,
                RightMatrix::EntryMinor: KeyValGet<RightMatrix::RowIndex, RightMatrix::Coefficient>,
                RightMatrix::ColIndex: Eq + Hash + Clone, 
                RightMatrix::Coefficient: Clone,
                RingOperator: Clone + Semiring<LeftMatrix::Coefficient>,
                OrderOperatorViewMajor: JudgePartialOrder<RightMatrix::ColIndex> + JudgePartialOrder<RightMatrix::EntryMajor> + Clone, 
                OrderOperatorKeyMajor: JudgePartialOrder<LeftMatrix::RowIndex> + JudgePartialOrder<LeftMatrix::EntryMinor> + Clone
{ 
    type EntryMajor = (Self::ColIndex, Self::Coefficient);
    type EntryMinor = (Self::RowIndex, Self::Coefficient);    
    type RowIndex = LeftMatrix::RowIndex;
    type ColIndex = RightMatrix::ColIndex; 
    type Coefficient = LeftMatrix::Coefficient;
}

/// Implement `ViewRowAscend` for `BimajorProductMatrix`
impl<'a, LeftMatrix, RightMatrix, RingOperator, OrderOperatorKeyMajor, OrderOperatorViewMajor>
    ViewRowAscend for 
        Arc<BimajorSparseProductMatrix<LeftMatrix, RightMatrix, RingOperator, OrderOperatorKeyMajor, OrderOperatorViewMajor>>
            where 
                LeftMatrix: ViewColDescend + ViewRowAscend + IndicesAndCoefficients,  
                LeftMatrix::ViewMajorAscend: IntoIterator,
                LeftMatrix::EntryMajor: KeyValGet<LeftMatrix::ColIndex, LeftMatrix::Coefficient>,
                LeftMatrix::EntryMinor: KeyValGet<LeftMatrix::RowIndex, LeftMatrix::Coefficient> + KeyValSet<LeftMatrix::RowIndex, LeftMatrix::Coefficient>,
                LeftMatrix::RowIndex: Eq + Hash + Clone,
                RightMatrix: ViewColDescend + ViewRowAscend + IndicesAndCoefficients<Coefficient = LeftMatrix::Coefficient, RowIndex = LeftMatrix::ColIndex>,  
                RightMatrix::ViewMajorAscend: IntoIterator, 
                RightMatrix::EntryMajor: KeyValGet<RightMatrix::ColIndex, RightMatrix::Coefficient> + KeyValSet<RightMatrix::ColIndex, RightMatrix::Coefficient>,
                RightMatrix::EntryMinor: KeyValGet<RightMatrix::RowIndex, RightMatrix::Coefficient>,
                RightMatrix::ColIndex: Eq + Hash + Clone + PartialEq, 
                RightMatrix::Coefficient: Clone,
                RingOperator: Clone + Semiring<LeftMatrix::Coefficient>,
                OrderOperatorViewMajor: JudgePartialOrder<RightMatrix::ColIndex> + JudgePartialOrder<RightMatrix::EntryMajor> + Clone, 
                OrderOperatorKeyMajor: JudgePartialOrder<LeftMatrix::RowIndex> + JudgePartialOrder<LeftMatrix::EntryMinor> + Clone
{
    type ViewMajorAscend = Vec<Self::EntryMajor>;
    type ViewMajorAscendIntoIter = std::vec::IntoIter<Self::EntryMajor>;
    ///
    /// Obtain a major view of the `BimajorProductMatirx`.
    /// 
    fn view_major_ascend(&self, index: Self::RowIndex) -> Self::ViewMajorAscend {
        BimajorSparseProductMatrix::view_major_ascend(&self, index)
    }
}

/// Implement `ViewColDescend` for `BimajorSparseProductMatrix` 
impl<'a, LeftMatrix, RightMatrix, RingOperator, OrderOperatorKeyMajor, OrderOperatorViewMajor>
    ViewColDescend for 
        Arc<BimajorSparseProductMatrix<LeftMatrix, RightMatrix, RingOperator, OrderOperatorKeyMajor, OrderOperatorViewMajor>>
            where 
                LeftMatrix: ViewColDescend + ViewRowAscend + IndicesAndCoefficients,  
                LeftMatrix::ViewMajorAscend: IntoIterator,
                LeftMatrix::EntryMajor: KeyValGet<LeftMatrix::ColIndex, LeftMatrix::Coefficient>,
                LeftMatrix::EntryMinor: KeyValGet<LeftMatrix::RowIndex, LeftMatrix::Coefficient> + KeyValSet<LeftMatrix::RowIndex, LeftMatrix::Coefficient>,
                LeftMatrix::RowIndex: Eq + Hash + Clone,
                RightMatrix: ViewColDescend + ViewRowAscend + IndicesAndCoefficients<Coefficient = LeftMatrix::Coefficient, RowIndex = LeftMatrix::ColIndex>,  
                RightMatrix::ViewMajorAscend: IntoIterator, 
                RightMatrix::EntryMajor: KeyValGet<RightMatrix::ColIndex, RightMatrix::Coefficient> + KeyValSet<RightMatrix::ColIndex, RightMatrix::Coefficient>,
                RightMatrix::EntryMinor: KeyValGet<RightMatrix::RowIndex, RightMatrix::Coefficient>,
                RightMatrix::ColIndex: Eq + Hash + Clone + PartialEq, 
                RightMatrix::Coefficient: Clone,
                RingOperator: Clone + Semiring<LeftMatrix::Coefficient>,
                OrderOperatorViewMajor: JudgePartialOrder<RightMatrix::ColIndex> + JudgePartialOrder<RightMatrix::EntryMajor> + Clone, 
                OrderOperatorKeyMajor: JudgePartialOrder<LeftMatrix::RowIndex> + JudgePartialOrder<LeftMatrix::EntryMinor> + Clone
{
    type ViewMinorDescend = Vec<Self::EntryMinor>;
    type ViewMinorDescendIntoIter = std::vec::IntoIter<Self::EntryMinor>;
    ///
    /// Obtain a minor view of the `BimajorProductMatrix`. 
    /// 
    fn view_minor_descend(&self, index: Self::ColIndex) -> Self::ViewMinorDescend {  
        BimajorSparseProductMatrix::view_minor_descend(&self, index)
    }
}

