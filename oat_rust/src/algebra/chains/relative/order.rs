//! 
//! Order operator structs for filtrations of simplices/chains that arise from a filtered pair of Vietoris-Rips complexes. 
//! 
//! Throughout this module we refer, in code and text, to a filtered pair of Vietoris-Rips complexes. We denote the pair as 
//! (K, K0), with K0 being the subcomplex. Both VR complexes are equipped with a scale filtration, which we respectively 
//! denote as F and G. For example, a simplex p of K which is never a member of K0 will satisfy G(p) = INF. 
//! 

use std::clone::Clone;
use std::fmt::Debug;
use std::cmp::Ordering;
use std::hash::Hash;
use std::sync::Arc;
use std::marker::PhantomData;
use core::option::Option;
use std::collections::HashMap;

use crate::algebra::chains::relative::traits::{FactorFromArc, VariableSortOracleKeys, SimplexDiameter}; 
use crate::algebra::rings::operator_traits::{Semiring, Ring, DivisionRing};
use crate::topology::simplicial::simplices::filtered::SimplexFiltered;
use crate::utilities::order::JudgePartialOrder; 
use crate::algebra::matrices::query::{IndicesAndCoefficients, MatrixEntry, ViewCol, ViewColDescend, ViewRowAscend};
use crate::algebra::matrices::operations::umatch::row_major::Umatch;
use crate::algebra::vectors::entries::{KeyValGet, KeyValNew, KeyValSet}; 

//  ===========================================================
//  FILTRATIONS OF SIMPLICES
//  ===========================================================





//  ===========================================================
//  Struct: OrderOperatorFullComplexFiltrationSimplices 
//  ===========================================================

#[derive(Clone, Debug)]
/// 
/// An order operator on simplices for the Vietoris-Rips filtration of the "full complex", denoted K. 
/// 
/// Order simplices according to the filtration F of the full complex K. This order operator will determine, 
/// given two simplices, which is born first in K. 
///
pub struct OrderOperatorFullComplexFiltrationSimplices<DissimilarityMatrix, Filtration>
    where 
        Filtration: Copy + Debug + PartialOrd + Ord,
        DissimilarityMatrix: IndicesAndCoefficients<ColIndex=usize, RowIndex=usize, Coefficient=Filtration> + ViewRowAscend + MatrixEntry
{
    /// A dissimilarity matrix for the full space data. 
    pub dissimilarity_matrix: DissimilarityMatrix, 
    /// Minimum dissimilarity threshold for simplices. 
    pub dissimilarity_min: Filtration
}

/// Implementation and methods for `OrderOperatorFullComplexFiltrationSimplices`
impl <DissimilarityMatrix, Filtration>
    OrderOperatorFullComplexFiltrationSimplices<DissimilarityMatrix, Filtration>
        where 
            Filtration: Copy + Debug + PartialOrd + Ord,
            DissimilarityMatrix: IndicesAndCoefficients<ColIndex=usize, RowIndex=usize, Coefficient=Filtration> + ViewRowAscend + MatrixEntry,
            DissimilarityMatrix::EntryMajor: KeyValGet<usize, Filtration> + Debug + Copy
{ 
    /// 
    /// Construct a `OrderOperatorFullComplexFiltrationSimplices` structure. 
    /// 
    pub fn new( 
        dissimilarity_matrix: DissimilarityMatrix,
        dissimilarity_min: Filtration
    ) -> Self 

    { 
        OrderOperatorFullComplexFiltrationSimplices { 
            dissimilarity_matrix,
            dissimilarity_min
        }
    }
}

/// Implement SimplexDiameter for `OrderOperatorFullComplexFiltrationSimplices`
impl <DissimilarityMatrix, Filtration> 
    SimplexDiameter<Filtration> for 
        OrderOperatorFullComplexFiltrationSimplices<DissimilarityMatrix, Filtration>
            where
                DissimilarityMatrix: IndicesAndCoefficients<ColIndex=usize, RowIndex=usize, Coefficient=Filtration> + ViewRowAscend + MatrixEntry,
                DissimilarityMatrix::EntryMajor: KeyValGet<usize, Filtration> + Debug + Copy,
                Filtration: Copy + Debug + PartialOrd + Ord
{ 

    /// 
    /// Compute diameter of a simplex using the full space dissimilarity matrix. 
    /// 
    /// Note: although this function has quadratic runtime, it is quadratic in 
    /// the number of provided `vertices`. Since the maximal simplex dimension of 
    /// `RelativeBoundaryMatrixOracle` is rarely more than 1 or 2, then the 
    /// number of vertices provided is rarely more than 2 or 3. Thus, quadratic 
    /// runtime is not an issue here. 
    /// 
    fn diameter(&self, vertices: &Vec<u16>) -> Option<Filtration> {
        let mut diameter = self.dissimilarity_min.clone(); 
        let mut a; 
        let mut b; 
        for i in 0..vertices.len() { 
            a = usize::from(vertices[i]); 
            for j in i..vertices.len() { 
                b = usize::from(vertices[j]); 
                match self.dissimilarity_matrix.entry_major_at_minor(a, b) {
                    None => { return None } 
                    Some( curr_diam ) => {
                        if curr_diam > diameter { diameter = curr_diam.clone(); }
                    }
                }
            }
        }
        return Some(diameter); 
    }

    ///
    /// Return the smallest structural nonzero entry of the full space dissimilarity matrix. 
    /// 
    fn dissimilarity_min(&self) -> Filtration {
        return self.dissimilarity_min.clone(); 
    }
}

/// Implement JudePartialOrder for `OrderOperatorFullComplexFiltrationSimplices`
impl <DissimilarityMatrix, Filtration> 
    JudgePartialOrder<SimplexFiltered<Filtration>> for
        OrderOperatorFullComplexFiltrationSimplices<DissimilarityMatrix, Filtration>
            where 
                DissimilarityMatrix: IndicesAndCoefficients<ColIndex=usize, RowIndex=usize, Coefficient=Filtration> + ViewRowAscend + MatrixEntry,
                DissimilarityMatrix::EntryMajor: KeyValGet<usize, Filtration> + Debug + Copy,
                Filtration: Copy + Debug + PartialOrd + Ord
{ 
    ///
    /// Detemrine if a given simplex `lhs` should appear before a simplex `rhs` 
    /// when moving left to right along the columns of the relative boundary 
    /// matrix. If this is true, then `Some(Ordering::Less)` is returned.
    /// 
    /// Simplices are compared first by diameter (filtration value) and then 
    /// lexicographically by comparing the vertices of each simplex. 
    /// 
    fn judge_partial_cmp(
        &self, 
        lhs: &SimplexFiltered<Filtration>, 
        rhs: &SimplexFiltered<Filtration>
    ) -> Option<Ordering>

    { 
        if self.judge_lt(lhs,rhs) { 
            return Some(Ordering::Less); 
        } else { 
            return Some(Ordering::Greater);
        }
    }

    ///
    /// Return true iff `lhs` < `rhs`, or in other words the provided simplex 
    /// `lhs` should appear before the provided simplex `rhs` when moving left
    /// to right along the columns of the relative boundary matrix. 
    /// 
    fn judge_lt(
        &self, 
        lhs: &SimplexFiltered<Filtration>, 
        rhs: &SimplexFiltered<Filtration>
    ) -> bool

    {
        let lhs_vertices = &lhs.vertices;
        let rhs_vertices = &rhs.vertices;
        let lhs_diameter = self.diameter(lhs_vertices);
        let rhs_diameter = self.diameter(rhs_vertices);
        // FIRST: compare diameter in K
        if lhs_diameter.is_none() || rhs_diameter.is_none() { 
            panic!("\n\nError: `OrderOperatorFullComplexFiltrationSimplices` failed to compare simplices {:?} and {:?}. One or both of these simplices never enters the full space filtration. \n This message is generated by OAT.\n\n", lhs, rhs);
        }
        else if lhs_diameter.unwrap() < rhs_diameter.unwrap() { 
            true 
        }
        else if lhs_diameter.unwrap() > rhs_diameter.unwrap() { 
            false 
        // lhs and rhs have same diameter
        // SECOND: compare vertices lexicographically 
        } else { 
            return lhs_vertices.cmp(rhs_vertices) == Ordering::Less
        }
    }

    ///
    /// Return true iff `lhs` > `rhs`, or in other words the provided simplex 
    /// `lhs` should appear after the provided simplex `rhs` when moving left
    /// to right along the columns of the relative boundary matrix. 
    /// 
    fn judge_gt(
        &self, 
        lhs: &SimplexFiltered<Filtration>, 
        rhs: &SimplexFiltered<Filtration>
    ) -> bool 
    
    { 
        let lhs_vertices = &lhs.vertices;
        let rhs_vertices = &rhs.vertices;
        let lhs_diameter = self.diameter(lhs_vertices);
        let rhs_diameter = self.diameter(rhs_vertices);
        // FIRST: compare diameter in K
        if lhs_diameter.is_none() || rhs_diameter.is_none() { 
            panic!("\n\nError: `OrderOperatorFullComplexFiltrationSimplices` failed to compare simplices {:?} and {:?}. One or both of these simplices never enters the full space filtration. \n This message is generated by OAT.\n\n", lhs, rhs);
        }
        else if lhs_diameter.unwrap() > rhs_diameter.unwrap() { 
            true 
        }
        else if lhs_diameter.unwrap() < rhs_diameter.unwrap() { 
            false 
        // lhs and rhs have same diameter
        // SECOND: compare vertices lexicographically 
        } else { 
            return lhs_vertices.cmp(rhs_vertices) == Ordering::Greater
        }
    }

    ///
    /// Return true iff `lhs` <= `rhs`, or in other words the provided simplex 
    /// `lhs` should appear at or before the provided simplex `rhs` when moving left
    /// to right along the columns of the relative boundary matrix. 
    /// 
    fn judge_le(
        &self, 
        lhs: &SimplexFiltered<Filtration>, 
        rhs: &SimplexFiltered<Filtration>
    ) -> bool  
    
    { 
        return !self.judge_gt(lhs, rhs); 
    }

    ///
    /// Return true iff `lhs` >= `rhs`, or in other words the provided simplex 
    /// `lhs` should appear at or after the provided simplex `rhs` when moving left
    /// to right along the columns of the relative boundary matrix. 
    /// 
    fn judge_ge(
        &self, 
        lhs: &SimplexFiltered<Filtration>, 
        rhs: &SimplexFiltered<Filtration>
    ) -> bool  
    
    { 
        return !self.judge_lt(lhs, rhs); 
    }
}

/// Alternative implementation of JudgePartialOrder for `OrderOperatorFullComplexFiltrationSimplices`
/// This implementation operates on entries (not keys) of a boundary oracle by comparing the keys of each entry. 
/// This is useful for constructing major or minor views whose order are inherited from keys. 
impl <DissimilarityMatrix, Filtration, Coefficient> 
    JudgePartialOrder<(SimplexFiltered<Filtration>, Coefficient)> for
        OrderOperatorFullComplexFiltrationSimplices<DissimilarityMatrix, Filtration>
            where 
                DissimilarityMatrix: IndicesAndCoefficients<ColIndex=usize, RowIndex=usize, Coefficient=Filtration> + ViewRowAscend + MatrixEntry,
                DissimilarityMatrix::EntryMajor: KeyValGet<usize, Filtration> + Debug + Copy,
                Filtration: Copy + Debug + PartialOrd + Ord,
                Coefficient: Clone + Debug + Hash
{ 
    fn judge_partial_cmp( 
        &self, 
        lhs: &(SimplexFiltered<Filtration>, Coefficient), 
        rhs: &(SimplexFiltered<Filtration>, Coefficient) 
    )-> Option<Ordering> 
    
    {
        return self.judge_partial_cmp(&lhs.0, &rhs.0); 
    }
}

//  ===========================================================
//  Struct: OrderOperatorSubComplexFiltrationSimplices
//  ===========================================================

#[derive(Clone, Debug)]
/// 
/// An order operator for the simplices of the "subcomplex", denoted K_0.  
/// 
/// Order simplices according to the filtration G of the subcomplex K0. This order operator 
/// will determine, given two simplices, which is born first in K0. 
///
pub struct OrderOperatorSubComplexFiltrationSimplices<DissimilarityMatrix, Filtration>
    where 
        DissimilarityMatrix: IndicesAndCoefficients<ColIndex=usize, RowIndex=usize, Coefficient=Filtration> + ViewRowAscend + MatrixEntry,
        Filtration: Copy + Debug + PartialOrd + Ord
{
    /// A dissimilarity matrix for the subspace data. 
    pub dissimilarity_matrix: DissimilarityMatrix,
    /// Minimum dissimilarity threshold for simplices. 
    pub dissimilarity_min: Filtration,
    /// A vector which maps data to custom subcomplex filtration values. 
    pub bijection_from_subcomplex_data_to_custom_filtration: Option<Vec<Option<Filtration>>> 
}

/// Implementation and methods for `OrderOperatorFullComplexFiltrationSimplices`
impl <DissimilarityMatrix, Filtration>
    OrderOperatorSubComplexFiltrationSimplices<DissimilarityMatrix, Filtration>
        where 
            DissimilarityMatrix: IndicesAndCoefficients<ColIndex=usize, RowIndex=usize, Coefficient=Filtration> + ViewRowAscend + MatrixEntry,
            DissimilarityMatrix::EntryMajor: KeyValGet<usize, Filtration> + Debug + Copy,
            Filtration: Copy + Debug + PartialOrd + Ord
{ 
    /// 
    /// Construct an `OrderOperatorSubComplexFiltrationSimplices` structure. 
    /// 
    pub fn new( 
        dissimilarity_matrix: DissimilarityMatrix,
        dissimilarity_min: Filtration, 
        dissimilarity_max: Filtration, 
        // NOTE to developers: although we do not offer the subcomplex filtration customization on the python side, it remains here 
        // as it is currently in development and in various unit tests for this module. 
        bijection_from_subcomplex_data_to_custom_filtration: Option< Vec<Option<Filtration>> > 
    ) -> Self 

    { 
        // safety checks; only needed if user provides customized subspace data
        if bijection_from_subcomplex_data_to_custom_filtration.is_some() { 
            let custom_data_vec = bijection_from_subcomplex_data_to_custom_filtration.clone().unwrap(); 
            for i in 0..custom_data_vec.clone().iter().count() { 
                let current_customized_entry = custom_data_vec[i]; 
                if current_customized_entry.is_some() {  
                    // ensure that customized data is in the subspace 
                    if dissimilarity_matrix.view_major_ascend(i).into_iter().count() == 0  { 
                        panic!("\n\nError: Constructing `OrderOperatorSubcomplexFiltrationSimplices` failed. Attempted to modify scale filtration for data not contained in the subspace. \n This message is generated by OAT.\n\n");
                    }
                } 
            }
        }
        // return struct
        OrderOperatorSubComplexFiltrationSimplices { 
            dissimilarity_matrix,
            dissimilarity_min, 
            bijection_from_subcomplex_data_to_custom_filtration
        }
    }
}

/// Implement SimplexDiameter for `OrderOperatorSubComplexFiltrationSimplices`
impl<DissimilarityMatrix, Filtration> 
    SimplexDiameter<Filtration> for 
        OrderOperatorSubComplexFiltrationSimplices<DissimilarityMatrix, Filtration>
            where
                DissimilarityMatrix: IndicesAndCoefficients<ColIndex=usize, RowIndex=usize, Coefficient=Filtration> + ViewRowAscend + MatrixEntry,
                DissimilarityMatrix::EntryMajor: KeyValGet<usize, Filtration> + Debug + Copy,
                Filtration: Copy + Debug + PartialOrd + Ord
{ 
    ///
    /// Compute the diameter of a simplex in the subcomplex using the associated dissimilarity matrix. 
    /// 
    /// Note: although this function has quadratic runtime, it is quadratic in 
    /// the number of provided `vertices`. Since the maximal simplex dimension of 
    /// `RelativeBoundaryMatrixOracle` is rarely more than 1 or 2, then the 
    /// number of vertices provided is rarely more than 2 or 3. Thus, quadratic 
    /// runtime is not an issue here. 
    fn diameter(&self, vertices: &Vec<u16>) -> Option<Filtration> {
        // compute diameter using the dissimilarity matrix
        let mut diameter_scale = self.dissimilarity_min.clone(); 
        let mut a; 
        let mut b; 
        for i in 0..vertices.len() { 
            a = usize::from( vertices[i] ); 
            for j in i..vertices.len() { 
                b = usize::from(vertices[j]); 
                match self.dissimilarity_matrix.entry_major_at_minor(a, b) {
                    None => { return None } 
                    Some( curr_diam ) => {
                        if curr_diam > diameter_scale { diameter_scale = curr_diam.clone(); }
                    }
                }
            }
        }
        // compute diameter using `self.bijection_from_subcomplex_data_to_custom_filtration`
        // NOTE: if we made it here, then we did not return `None` in the loop above ... thus the simplex given by `vertices` IS in the subcomplex 
        // now we check if it enters the filtration after it is born in the full complex! 
        let mut diameter_custom = None; 
        if self.bijection_from_subcomplex_data_to_custom_filtration.is_some() { 
            diameter_custom = vertices
                .iter()
                .filter_map(|&i| self.bijection_from_subcomplex_data_to_custom_filtration.clone().unwrap()[usize::from(i)])
                .max();
        }
        // compare scale and custom diameter and return 
        let diameter = match diameter_custom.is_some() { 
            true => { std::cmp::max(diameter_scale, diameter_custom.unwrap()) }, 
            false => { diameter_scale }
        }; 
        return Some(diameter); 
    }

    ///
    /// Return the smallest structural nonzero entry of the subspace dissimilarity matrix. 
    /// 
    fn dissimilarity_min(&self) -> Filtration {
        return self.dissimilarity_min.clone(); 
    }
}

/// Implement JudePartialOrder for `OrderOperatorSubComplexFiltrationSimplices`
impl <DissimilarityMatrix, Filtration> 
    JudgePartialOrder<SimplexFiltered<Filtration>> for
        OrderOperatorSubComplexFiltrationSimplices<DissimilarityMatrix, Filtration>
            where 
                DissimilarityMatrix: IndicesAndCoefficients<ColIndex=usize, RowIndex=usize, Coefficient=Filtration> + ViewRowAscend + MatrixEntry,
                DissimilarityMatrix::EntryMajor: KeyValGet<usize, Filtration> + Debug + Copy,
                Filtration: Copy + Debug + PartialOrd + Ord
{ 
    ///
    /// Detemrine if a given simplex `lhs` should appear before a simplex `rhs` 
    /// in the filtration on the subspace K0. If this is true, then 
    /// `Some(Ordering::Less)` is returned.
    /// 
    /// Simplices are compared first by diameter (or filtration value) and then 
    /// lexicographically by comparing the vertices of each simplex. 
    /// 
    fn judge_partial_cmp(
        &self, 
        lhs: &SimplexFiltered<Filtration>, 
        rhs: &SimplexFiltered<Filtration>
    ) -> Option<Ordering>

    { 
        if self.judge_lt(lhs,rhs) { 
            return Some(Ordering::Less); 
        } else { 
            return Some(Ordering::Greater);
        }
    }

    ///
    /// Return true iff `lhs` < `rhs`, or in other words the provided simplex 
    /// `lhs` should appear before the provided simplex `rhs` in the filtration 
    /// on the subspace, K0. 
    /// 
    fn judge_lt(
        &self, 
        lhs: &SimplexFiltered<Filtration>, 
        rhs: &SimplexFiltered<Filtration>
    ) -> bool

    {
        let lhs_vertices = &lhs.vertices;
        let rhs_vertices = &rhs.vertices;
        let lhs_diameter = self.diameter(lhs_vertices);
        let rhs_diameter = self.diameter(rhs_vertices);
        // FIRST: compare diameter in K0
        if lhs_diameter.is_none() || rhs_diameter.is_none() { 
            panic!("\n\nError: `OrderOperatorSubComplexFiltrationSimplices` failed to compare simplices {:?} and {:?}. One or both of these simplices never enters the subspace filtration. \n This message is generated by OAT.\n\n", lhs, rhs);
        }
        else if lhs_diameter.unwrap() < rhs_diameter.unwrap() { 
            true 
        }
        else if lhs_diameter.unwrap() > rhs_diameter.unwrap() { 
            false 
        // lhs and rhs have same diameter
        // SECOND: compare vertices lexicographically 
        } else { 
            return lhs_vertices.cmp(rhs_vertices) == Ordering::Less
        }
    }

    ///
    /// Return true iff `lhs` > `rhs`, or in other words the provided simplex 
    /// `lhs` should appear after the provided simplex `rhs` in the filtration 
    /// on the subspace, K0.
    /// 
    fn judge_gt(
        &self, 
        lhs: &SimplexFiltered<Filtration>, 
        rhs: &SimplexFiltered<Filtration>
    ) -> bool 
    
    { 
        let lhs_vertices = &lhs.vertices;
        let rhs_vertices = &rhs.vertices;
        let lhs_diameter = self.diameter(lhs_vertices);
        let rhs_diameter = self.diameter(rhs_vertices);
        // FIRST: compare diameter in K0
        if lhs_diameter.is_none() || rhs_diameter.is_none() { 
            panic!("\n\nError: `OrderOperatorSubComplexFiltrationSimplices` failed to compare simplices {:?} and {:?}. One or both of these simplices never enters the subspace filtration. \n This message is generated by OAT.\n\n", lhs, rhs);
        }
        else if lhs_diameter.unwrap() > rhs_diameter.unwrap() { 
            true 
        }
        else if lhs_diameter.unwrap() < rhs_diameter.unwrap() { 
            false 
        // lhs and rhs have same diameter
        // SECOND: compare vertices lexicographically 
        } else { 
            return lhs_vertices.cmp(rhs_vertices) == Ordering::Greater
        }
    }

    ///
    /// Return true iff `lhs` <= `rhs`, or in other words the provided simplex 
    /// `lhs` should appear at or before the provided simplex `rhs` in the filtration 
    /// on the subspace, K0. 
    /// 
    fn judge_le(
        &self, 
        lhs: &SimplexFiltered<Filtration>, 
        rhs: &SimplexFiltered<Filtration>
    ) -> bool  
    
    { 
        return !self.judge_gt(lhs, rhs); 
    }

    ///
    /// Return true iff `lhs` >= `rhs`, or in other words the provided simplex 
    /// `lhs` should appear at or after the provided simplex `rhs` in the filtration 
    /// on the subspace, K0.
    fn judge_ge(
        &self, 
        lhs: &SimplexFiltered<Filtration>, 
        rhs: &SimplexFiltered<Filtration>
    ) -> bool  
    
    { 
        return !self.judge_lt(lhs, rhs); 
    }
}
//  ===========================================================
//  Struct: RelativeBoundaryMatrixRowIndexOrderOperator
//  ===========================================================

#[derive(Clone, Debug)]
/// 
/// An order operator for the row indices of a `RelativeBoundaryMatrixOracle`.
/// 
/// Suppose that the ith row of the boundary matrix is indexed by the simplex p and that likewise the jth row is indexed by 
/// simplex q. This order operator declares that i <= j, or the ith row of the boundary matrix appears before the jth row of 
/// the boundary matrix in the relative boundary matrix, if and only if one of the following conditions hold: 
/// 
/// (i) p and q both lie in K_0 and G(p) <= G(q)
/// (ii) p and q both lie in K \ K_0 and F(p) <= F(q)
/// (iii) p lies in K_0 and q lies in K
/// 
/// Equivalently, place all simplices in K0 before all simplices in K \ K0, forming two disjoint sets of simplices along the 
/// rows of the relative boundary matrix with filtration value increasing with increasing row index. For the simplices in K0 we 
/// use the filtration G to compare, and for the simplices in K \ K0 we use the filtration F. 
/// 
pub struct RelativeBoundaryMatrixRowIndexOrderOperator<Filtration, OrderOperatorFullComplex, OrderOperatorSubComplex>
    where 
        Filtration: Copy + Debug + PartialOrd + Ord,
        OrderOperatorFullComplex: JudgePartialOrder<SimplexFiltered<Filtration>> + SimplexDiameter<Filtration>, 
        OrderOperatorSubComplex: JudgePartialOrder<SimplexFiltered<Filtration>> + SimplexDiameter<Filtration>
{
    // phantom data 
    phantom_filtration: PhantomData<Filtration>, 

    /// The order operator for the filtration F on the full complex K, or the full space filtration. 
    pub order_operator_full_space: OrderOperatorFullComplex,
    /// The order operator for the filtration G on the sub complex K0, or the subspace filtration. 
    pub order_operator_subspace: OrderOperatorSubComplex
}

/// Implementation and methods for `RelativeBoundaryMatrixRowIndexOrderOperator`
impl <Filtration, OrderOperatorFullComplex, OrderOperatorSubComplex>
    RelativeBoundaryMatrixRowIndexOrderOperator<Filtration, OrderOperatorFullComplex, OrderOperatorSubComplex>
        where 
            Filtration: Copy + Debug + PartialOrd + Ord,
            OrderOperatorFullComplex: JudgePartialOrder<SimplexFiltered<Filtration>> + SimplexDiameter<Filtration>, 
            OrderOperatorSubComplex: JudgePartialOrder<SimplexFiltered<Filtration>> + SimplexDiameter<Filtration>
{ 
    ///
    /// Construct a `RelativeBoundaryMatrixRowIndexOrderOperator` structure. 
    /// Must provide a filtered VR complex for K and K0.
    ///  
    pub fn new(
        order_operator_full_space: OrderOperatorFullComplex, 
        order_operator_subspace: OrderOperatorSubComplex
    ) -> Self

    { 
        RelativeBoundaryMatrixRowIndexOrderOperator { 
            phantom_filtration: PhantomData, 
            order_operator_full_space, 
            order_operator_subspace
        }
    }

    ///
    /// Given a simplex generated by data of the full space, determine if the simplex 
    /// ever enters the filtration of the subspace. 
    /// 
    /// Panics in the case that the provided `simplex` never enters the filtration of the full space. 
    /// 
    pub fn is_subspace_simplex(&self, simplex: &SimplexFiltered<Filtration>) -> bool { 
        return self.order_operator_subspace.diameter(&simplex.vertices).is_some(); 
    }
}

/// Implement JudePartialOrder for `RelativeBoundaryMatrixRowIndexOrderOperator`
impl <Filtration, OrderOperatorFullComplex, OrderOperatorSubComplex> 
    JudgePartialOrder<SimplexFiltered<Filtration>> for
        RelativeBoundaryMatrixRowIndexOrderOperator<Filtration, OrderOperatorFullComplex, OrderOperatorSubComplex>    
            where 
                Filtration: Copy + Debug + PartialOrd + Ord, 
                OrderOperatorFullComplex: JudgePartialOrder<SimplexFiltered<Filtration>> + SimplexDiameter<Filtration>, 
                OrderOperatorSubComplex: JudgePartialOrder<SimplexFiltered<Filtration>> + SimplexDiameter<Filtration>
{ 
    /// 
    /// Detemrine if a given simplex `lhs` should appear before a simplex `rhs` 
    /// when moving down the rows of the relative boundary matrix. If this is 
    /// true, then `Some(Ordering::Less)` is returned. 
    /// 
    /// Simplices are first compared by mebership in subspace K0, then by diameter
    /// (can also use filtration value) then lexicographically by comparing vertices 
    /// of each simplex. 
    /// 
    fn judge_partial_cmp(
        &self, 
        lhs: &SimplexFiltered<Filtration>, 
        rhs: &SimplexFiltered<Filtration>
    ) -> Option<Ordering> 

    { 
        if self.judge_lt(lhs,rhs) { 
            return Some(Ordering::Less); 
        } else { 
            return Some(Ordering::Greater);
        }
    }

    /// 
    /// Return true iff `lhs` < `rhs`, or in other words the provided simplex 
    /// `lhs` should appear before the provided simplex `rhs` when moving down 
    /// the rows of the relative boundary matrix.
    /// 
    fn judge_lt(
        &self, 
        lhs: &SimplexFiltered<Filtration>, 
        rhs: &SimplexFiltered<Filtration>
    ) -> bool
    
    {    
        // determine subspace (K0) membership of lhs and rhs
        let lhs_is_subspace_member = self.is_subspace_simplex(lhs); 
        let rhs_is_subspace_member = self.is_subspace_simplex(rhs); 
        // FIRST: compare membership in K0
        // lhs in K0 and rhs not in K0
        if lhs_is_subspace_member && !rhs_is_subspace_member { 
            return true; 
        }
        // lhs not in K0 and rhs in K0
        else if !lhs_is_subspace_member && rhs_is_subspace_member { 
            return false; 
        }
        // lhs and rhs both in K0 
        // SECOND: compare diameter/filtration with `OrderOperatorSubComplexfiltrationSimplices`
        else if lhs_is_subspace_member && rhs_is_subspace_member { 
            return self.order_operator_subspace.judge_lt(lhs, rhs); 
        }
        // lhs and rhs both not in K0
        // SECOND: compare diameter/filtration with `OrderOperatorFullComplexfiltrationSimplices`
        else { 
            return self.order_operator_full_space.judge_lt(lhs, rhs);
        }
    }

    /// 
    /// Return true iff `lhs` > `rhs`, or in other words the provided simplex 
    /// `lhs` should appear after the provided simplex `rhs` when moving down 
    /// the rows of the relative boundary matrix.
    /// 
    fn judge_gt(
        &self, 
        lhs: &SimplexFiltered<Filtration>, 
        rhs: &SimplexFiltered<Filtration>
    ) -> bool

    { 
        // determine subspace (K0) membership of lhs and rhs
        let lhs_is_subspace_member = self.is_subspace_simplex(lhs); 
        let rhs_is_subspace_member = self.is_subspace_simplex(rhs); 
        // FIRST: compare membership in K0
        // lhs in K0 and rhs not in K0
        if lhs_is_subspace_member && !rhs_is_subspace_member { 
            return false; 
        }
        // lhs not in K0 and rhs in K0
        else if !lhs_is_subspace_member && rhs_is_subspace_member { 
            return true; 
        }
        // lhs and rhs both in K0 
        // SECOND: compare diameter/filtration with `OrderOperatorSubComplexfiltrationSimplices`
        else if lhs_is_subspace_member && rhs_is_subspace_member { 
            return self.order_operator_subspace.judge_gt(lhs, rhs); 
        }
        // lhs and rhs both not in K0
        // SECOND: compare diameter/filtration with `OrderOperatorFullComplexfiltrationSimplices`
        else { 
            return self.order_operator_full_space.judge_gt(lhs, rhs);
        }
    }

    /// 
    /// Return true iff `lhs` <= `rhs`, or in other words the provided simplex 
    /// `lhs` should appear at or before the provided simplex `rhs` when moving down 
    /// the rows of the relative boundary matrix.
    /// 
    fn judge_le(
        &self, 
        lhs: &SimplexFiltered<Filtration>, 
        rhs: &SimplexFiltered<Filtration>
    ) -> bool 
    
    { 
        return !self.judge_gt(lhs, rhs); 
    }

    /// 
    /// Return true iff `lhs` >= `rhs`, or in other words the provided simplex 
    /// `lhs` should appear at or after the provided simplex `rhs` when moving down 
    /// the rows of the relative boundary matrix.
    /// 
    fn judge_ge(
        &self, 
        lhs: &SimplexFiltered<Filtration>, 
        rhs: &SimplexFiltered<Filtration>
    ) -> bool 
    
    { 
        return !self.judge_lt(lhs, rhs); 
    }
}

/// Alternative implementation of JudgePartialOrder for `RelativeBoundaryMatrixRowIndexOrderOperator`
/// This implementation operates on entries (not keys) of a boundary oracle by comparing the keys of each entry. 
/// This is useful for constructing major or minor views whose order are inherited from keys. 
impl <Filtration, Coefficient, OrderOperatorFullComplex, OrderOperatorSubComplex> 
    JudgePartialOrder<(SimplexFiltered<Filtration>, Coefficient)> for
        RelativeBoundaryMatrixRowIndexOrderOperator<Filtration, OrderOperatorFullComplex, OrderOperatorSubComplex>    
            where 
                Filtration: Copy + Debug + PartialOrd + Ord, 
                OrderOperatorFullComplex: JudgePartialOrder<SimplexFiltered<Filtration>> + SimplexDiameter<Filtration>, 
                OrderOperatorSubComplex: JudgePartialOrder<SimplexFiltered<Filtration>> + SimplexDiameter<Filtration>
{ 
    fn judge_partial_cmp( 
        &self, 
        lhs: &(SimplexFiltered<Filtration>, Coefficient), 
        rhs: &(SimplexFiltered<Filtration>, Coefficient) 
    ) -> Option<Ordering> 
    
    {
        return self.judge_partial_cmp(&lhs.0, &rhs.0); 
    }
}





//  ===========================================================
//  FILTRATIONS OF RELATIVE CHAINS
//  ===========================================================





//  ===========================================================
//  Struct: SourceColumnIndexOrderOperator
//  ===========================================================

#[derive(Clone, Debug)]
/// 
/// Given the source (domain) COMB for the U-Match of the boundary oracle of a filtered quotient, chain complex, this struct
/// provides methods for determining if/when a relative chain along the columns of this COMB is born as a relative cycle. 
/// Implements traits for placing a total order on the minor keys of this COMB according relative cycle birth of the 
/// corresponding minor views. 
/// 
/// Place a total order on the column indices (minor keys) of the source COMB matrix oracle of the U-Match decomposition of 
/// a `RelativeBoundaryMatrixOracle`. We refer to the U-Match simpy as TM = DS where D is a `RelativeBoundaryMatrixOracle`
/// and S is the source COMB matrix oracle. The minor keys of the source COMB are the same simplices, or `SimplexFiltered` 
/// structs, which are the minor keys of the associated `RelativeBoundaryMatrixOracle`. Thus, this ordering on these simplices 
/// is not according to the birth of the indexing simplices themselves, but rather the birth of the chains along the columns 
/// they index as relative cycles. 
/// 
/// This ordering on the columns of S allows for: 
/// 
/// (i) the construction of a matrix B which contains a filtered basis for relative cycles. 
/// 
/// (ii) a second U-Match to be performed which "stitches" together one filtered basis containing all relative cycles and 
/// relative boundaries within its span. 
/// 
/// The birth of the chain x as a relative cycle in the filtered quotient space K/K0 is given by max(a,b) where a = F(x) 
/// and b = G(Dx). In words, a is the birth of x in the filtration of the full complex and b is the birth of the boundary 
/// of x in the filtration of the sub complex. Equivalently, since we use max(a,b), then the birth of a chain as a relative 
/// cycle is the first time step at which the chain exists in the full complex K and has a "trivial" boundary in the quotient 
/// topology. 
/// 
/// Suppose that the ith column of S is indexed by the chain p and that likewise the jth column is indexed by the chain q. 
/// Let b(p) and b(q) denote their birth as relative cycles computed as noted above. This order operator declares that the ith 
/// column of S appears before the jth column of S in the filtered basis B if and only if b(p) <= b(q).  
/// 
/// Any chain x never born as a realtive cycle will be assigned b(x) = `None`. These chains are then ordered by lexicographically 
/// comparing the corresponding indexing simplices. 
/// 
pub struct SourceColumnIndexOrderOperator<
    RelativeOracle, Filtration, Coefficient, RingOperator, 
    OrderOperatorOracleKeyMinor, OrderOperatorOracleKeyMajor, 
    OrderOperatorOracleViewMinor, OrderOperatorOracleViewMajor, 
    OrderOperatorSubComplex>
        where 
            // trait bounds are identical of those to `RelativeBoundaryMatrixOracleWrapper`
            RelativeOracle: Clone + VariableSortOracleKeys<SimplexFiltered<Filtration>, OrderOperatorOracleKeyMajor, OrderOperatorOracleKeyMinor, OrderOperatorSubComplex> + FactorFromArc<RelativeOracle, RingOperator, OrderOperatorOracleKeyMinor, OrderOperatorOracleKeyMajor>,             
            Arc<RelativeOracle>: ViewRowAscend + ViewColDescend + IndicesAndCoefficients<RowIndex=SimplexFiltered<Filtration>, ColIndex=SimplexFiltered<Filtration>, Coefficient=Coefficient, EntryMajor=(SimplexFiltered<Filtration>, Coefficient), EntryMinor=(SimplexFiltered<Filtration>, Coefficient)>, 
            <Arc<RelativeOracle> as IndicesAndCoefficients>::EntryMajor: Clone + KeyValSet<SimplexFiltered<Filtration>, Coefficient> + KeyValNew<SimplexFiltered<Filtration>, Coefficient> + KeyValGet<SimplexFiltered<Filtration>, Coefficient>, 
            <Arc<RelativeOracle> as IndicesAndCoefficients>::EntryMinor: Clone + KeyValSet<SimplexFiltered<Filtration>, Coefficient> + KeyValNew<SimplexFiltered<Filtration>, Coefficient> + KeyValGet<SimplexFiltered<Filtration>, Coefficient>,
            OrderOperatorOracleKeyMinor: JudgePartialOrder<SimplexFiltered<Filtration>> + SimplexDiameter<Filtration>,    
            OrderOperatorSubComplex: JudgePartialOrder<SimplexFiltered<Filtration>> + SimplexDiameter<Filtration>,        
            OrderOperatorOracleKeyMajor: JudgePartialOrder<SimplexFiltered<Filtration>>,                          
            OrderOperatorOracleViewMajor: Clone + JudgePartialOrder<(SimplexFiltered<Filtration>, Coefficient)>, 
            OrderOperatorOracleViewMinor: Clone + JudgePartialOrder<(SimplexFiltered<Filtration>, Coefficient)>,
            Filtration: Clone + Copy + Debug + PartialOrd + Ord + Hash,
            Coefficient: Clone + Debug + Hash,
            RingOperator: Semiring<Coefficient> + Ring<Coefficient> + DivisionRing<Coefficient> + Copy + Debug
{
    /// The U-Match deomposition of a `RelativeBoundaryMatrixOracle`.
    relative_umatch: Umatch<Arc<RelativeOracle>, RingOperator, OrderOperatorOracleViewMajor, OrderOperatorOracleViewMinor>, 
    /// An order operator comparing simplices in the filtration G on the sub complex K0. 
    order_operator_sub_complex_filtration: OrderOperatorSubComplex,           
    /// An order operator comparing simplices in the filtration F on the full complex K.
    order_operator_full_complex_filtration: OrderOperatorOracleKeyMinor, 
    /// An order operator for the major keys of the factored `RelativeOracle`
    order_operator_oracle_key_major_phantom: PhantomData<OrderOperatorOracleKeyMajor>     
}  

/// Implementation and methods for `SourceColumnIndexOrderOperator`
impl<RelativeOracle, Filtration, Coefficient, RingOperator, OrderOperatorOracleKeyMinor, OrderOperatorOracleKeyMajor, OrderOperatorOracleViewMinor, OrderOperatorOracleViewMajor, OrderOperatorSubComplex>
    SourceColumnIndexOrderOperator<RelativeOracle, Filtration, Coefficient, RingOperator, OrderOperatorOracleKeyMinor, OrderOperatorOracleKeyMajor, OrderOperatorOracleViewMinor, OrderOperatorOracleViewMajor, OrderOperatorSubComplex>
        where 
            RelativeOracle: Clone + VariableSortOracleKeys<SimplexFiltered<Filtration>, OrderOperatorOracleKeyMajor, OrderOperatorOracleKeyMinor, OrderOperatorSubComplex> + FactorFromArc<RelativeOracle, RingOperator, OrderOperatorOracleKeyMinor, OrderOperatorOracleKeyMajor>,             
            Arc<RelativeOracle>: ViewRowAscend + ViewColDescend + IndicesAndCoefficients<RowIndex=SimplexFiltered<Filtration>, ColIndex=SimplexFiltered<Filtration>, Coefficient=Coefficient, EntryMajor=(SimplexFiltered<Filtration>, Coefficient), EntryMinor=(SimplexFiltered<Filtration>, Coefficient)>, 
            <Arc<RelativeOracle> as IndicesAndCoefficients>::EntryMajor: Clone + KeyValSet<SimplexFiltered<Filtration>, Coefficient> + KeyValNew<SimplexFiltered<Filtration>, Coefficient> + KeyValGet<SimplexFiltered<Filtration>, Coefficient>, 
            <Arc<RelativeOracle> as IndicesAndCoefficients>::EntryMinor: Clone + KeyValSet<SimplexFiltered<Filtration>, Coefficient> + KeyValNew<SimplexFiltered<Filtration>, Coefficient> + KeyValGet<SimplexFiltered<Filtration>, Coefficient>,
            OrderOperatorOracleKeyMinor: JudgePartialOrder<SimplexFiltered<Filtration>> + SimplexDiameter<Filtration>,    
            OrderOperatorSubComplex: JudgePartialOrder<SimplexFiltered<Filtration>> + SimplexDiameter<Filtration>,        
            OrderOperatorOracleKeyMajor: JudgePartialOrder<SimplexFiltered<Filtration>>,                          
            OrderOperatorOracleViewMajor: Clone + JudgePartialOrder<(SimplexFiltered<Filtration>, Coefficient)>, 
            OrderOperatorOracleViewMinor: Clone + JudgePartialOrder<(SimplexFiltered<Filtration>, Coefficient)>,
            Filtration: Clone + Copy + Debug + PartialOrd + Ord + Hash,
            Coefficient: Clone + Debug + Hash,
            RingOperator: Semiring<Coefficient> + Ring<Coefficient> + DivisionRing<Coefficient> + Copy + Debug
{
    /// 
    /// Construct a `SourceColumnIndexOrderOperator` structure. 
    /// 
    /// User must provide the U-Match decomposition of a relative boundary matrix and 
    /// the order operators on the filtration of the full and sub complexes associated 
    /// with the relative boundary matrix. 
    /// 
    pub fn new(
        relative_umatch: Umatch<Arc<RelativeOracle>, RingOperator, OrderOperatorOracleViewMajor, OrderOperatorOracleViewMinor>,  
        order_operator_sub_complex_filtration: OrderOperatorSubComplex,           
        order_operator_full_complex_filtration: OrderOperatorOracleKeyMinor 
    ) -> Self
    
    {         
        SourceColumnIndexOrderOperator { 
            relative_umatch, 
            order_operator_sub_complex_filtration, 
            order_operator_full_complex_filtration, 
            order_operator_oracle_key_major_phantom: PhantomData
        }
    } 

    /// 
    /// Returns the filtration parameter where column S[:,col_index] of the 
    /// source COMB is born as a relative cycle. Note that `col_index` refers
    /// to a simplex which is a column index (minor key) of both S and the 
    /// associated `RelativeBoundaryMatrixOracle` D. In the case that this 
    /// column vector/chain is not a relative cycle, then `None` is returned. 
    /// 
    pub fn relative_cycle_birth(
        &self, 
        col_index: &SimplexFiltered<Filtration>
    ) -> Option<Filtration>
        
    { 
        // STEP 1: Determine the birth of the chain given by S[:,col_index] in the full complex filtration F
        // =======================================================================================================================================
        // - `source_minor_view` is the column vector/chain given by S[:,col_index], which is just a linear combination of simplices
        // - thus, the birth of S[:,col_index] in F is the first time step at which all simplices of the linear combination exist in F 
        // - in OAT, the minor view S[:,col_index] will contain all entries sorted by descending (bottom to top) order of index
        // - the major keys of S are also the major keys of D, so this particular minor view is just an iterator over simplices and coefficients 
        // - this filtrtion vlaue is given by `_a`.    
        // =======================================================================================================================================
        let source_minor_view: Vec<(SimplexFiltered<Filtration>, Coefficient)> = self.relative_umatch.comb_domain().view_minor_descend(col_index.clone()).collect();  
        let mut _a = None; 
        for entry in source_minor_view.into_iter() {     
            let birth = self.order_operator_full_complex_filtration.diameter(&entry.0.vertices);
            if birth.is_some() { 
                if _a.is_some() { 
                    _a = std::cmp::max(_a,birth); 
                } 
                else { 
                    _a = birth; 
                }
            }
        }

        // STEP 2: Use M[:,col_index] to determine the birth of the boundary of the chain given by S[:,col_index] in the sub complex filtration G
        // =======================================================================================================================================
        // - `matching_minor_view` is the column vector given by M[:,col_index]
        // - since `matching_minor_view` contains at most one nonzero coefficient, then we get the birth of the corresponding major key if necessary
        // - in the case that `matching_minor_view` is the zero vector, we have an absolute (and thus relative) cycle with `_b` = `dissimilarity_min` 
        // =======================================================================================================================================
        let matching_minor_view = self.relative_umatch.matching_ref().view_minor(col_index.clone());  
        let mut _b = None;  
        if matching_minor_view.clone().count() == 0 { 
            _b = Some(self.order_operator_sub_complex_filtration.dissimilarity_min());
        }
        else {  
            let entry_vertices = matching_minor_view.clone().last().unwrap().key().vertices; 
            _b = self.order_operator_sub_complex_filtration.diameter(&entry_vertices)
        }

        // STEP 3: Return the maximum of the filtration values `_a`` and `_b`
        // =======================================================================================================================================
        // - The birth of a chain as a relative cycle is the first time in which both of the following conditions hold: 
        //      (i.) the chain has been born in the filtration F on K 
        //      (ii.) the boundary of the chain has been born in the filtration G on K0
        // - In the case both of these conditions hold, then S[:,col_index] is a chain with "trivial" boundary in the quotient topology. 
        // - Note that if S[:,col_index] is an absolute cycle then (ii.) is true trivially. 
        // =======================================================================================================================================
        if _a.is_none() && _b.is_none() { 
            return None; 
        }
        else if _a.is_some() && _b.is_none() {
            return _b;
        }
        else if _a.is_none() && _b.is_some() { 
            return _a; 
        }
        else { 
            return std::cmp::max(_a,_b); 
        }
    }

    ///
    /// Determine if a given chain, or column vector of S, is a 
    /// relative cycle. The user must provide the `SimplexFiltered` 
    /// struct indexing that column. 
    /// 
    /// This function simply wraps a call to `self.relative_cycle_birth`. 
    /// 
    pub fn is_relative_cycle(
        &self, 
        col_index: &SimplexFiltered<Filtration>
    ) -> bool

    { 
        return self.relative_cycle_birth(col_index).is_some(); 
    }
}

/// Implement JudgePartialOrder for `SourceColumnIndexOrderOperator`
impl<RelativeOracle, Filtration, Coefficient, RingOperator, OrderOperatorOracleKeyMinor, OrderOperatorOracleKeyMajor, OrderOperatorOracleViewMinor, OrderOperatorOracleViewMajor, OrderOperatorSubComplex>
    JudgePartialOrder<SimplexFiltered<Filtration>> for 
        SourceColumnIndexOrderOperator<RelativeOracle, Filtration, Coefficient, RingOperator, OrderOperatorOracleKeyMinor, OrderOperatorOracleKeyMajor, OrderOperatorOracleViewMinor, OrderOperatorOracleViewMajor, OrderOperatorSubComplex>
            where 
                RelativeOracle: Clone + VariableSortOracleKeys<SimplexFiltered<Filtration>, OrderOperatorOracleKeyMajor, OrderOperatorOracleKeyMinor, OrderOperatorSubComplex> + FactorFromArc<RelativeOracle, RingOperator, OrderOperatorOracleKeyMinor, OrderOperatorOracleKeyMajor>,             
                Arc<RelativeOracle>: ViewRowAscend + ViewColDescend + IndicesAndCoefficients<RowIndex=SimplexFiltered<Filtration>, ColIndex=SimplexFiltered<Filtration>, Coefficient=Coefficient, EntryMajor=(SimplexFiltered<Filtration>, Coefficient), EntryMinor=(SimplexFiltered<Filtration>, Coefficient)>, 
                <Arc<RelativeOracle> as IndicesAndCoefficients>::EntryMajor: Clone + KeyValSet<SimplexFiltered<Filtration>, Coefficient> + KeyValNew<SimplexFiltered<Filtration>, Coefficient> + KeyValGet<SimplexFiltered<Filtration>, Coefficient>, 
                <Arc<RelativeOracle> as IndicesAndCoefficients>::EntryMinor: Clone + KeyValSet<SimplexFiltered<Filtration>, Coefficient> + KeyValNew<SimplexFiltered<Filtration>, Coefficient> + KeyValGet<SimplexFiltered<Filtration>, Coefficient>,
                OrderOperatorOracleKeyMinor: JudgePartialOrder<SimplexFiltered<Filtration>> + SimplexDiameter<Filtration>, 
                OrderOperatorSubComplex: JudgePartialOrder<SimplexFiltered<Filtration>> + SimplexDiameter<Filtration>,        
                OrderOperatorOracleKeyMajor: JudgePartialOrder<SimplexFiltered<Filtration>>,                          
                OrderOperatorOracleViewMajor: Clone + JudgePartialOrder<(SimplexFiltered<Filtration>, Coefficient)>, 
                OrderOperatorOracleViewMinor: Clone + JudgePartialOrder<(SimplexFiltered<Filtration>, Coefficient)>,
                Filtration: Clone + Copy + Debug + PartialOrd + Ord + Hash,
                Coefficient: Clone + Debug + Hash,
                RingOperator: Semiring<Coefficient> + Ring<Coefficient> + DivisionRing<Coefficient> + Copy + Debug
{
    /// 
    /// Detemrine if a given chain `lhs` should appear before a chain `rhs` in 
    /// the filtration on relative cycles along the columns of the source COMB. 
    /// If this is true, then `Some(Ordering::Less)` is returned. The user must 
    /// provide the `SimplexFiltered` structs which index the columns of S to 
    /// be compared. 
    /// 
    /// This function simply wraps a call to `self.judge_lt()`.
    /// 
    fn judge_partial_cmp(
        &self, 
        lhs: &SimplexFiltered<Filtration>, 
        rhs: &SimplexFiltered<Filtration>
    ) -> Option<Ordering>

    { 
        if self.judge_lt(lhs,rhs) { 
            return Some(Ordering::Less); 
        } else { 
            return Some(Ordering::Greater);
        }
    }

    ///
    /// Return true iff `lhs` < `rhs`. In words, the chain/column vector 
    /// corresponding to the provided minor key `lhs` should appear before the 
    /// chain/column vector corresponding to the provided minor key `rhs` in 
    /// the filtration of relative cycles along the columns of the source COMB. 
    /// 
    /// Chains are first judged by birth as a relative cycle using 
    /// `self.relative_cycle_birth()` and then by lexicographically comparing 
    /// indexing simplices (minor keys) in the case that either: 
    /// 
    /// (i.) two chains are born as relative cycles at the same filtration/diameter
    /// 
    /// (ii.) two chains are both never relative cycles, and have `None` as filtration 
    /// 
    /// The user must provide `SimplexFiltered` structs for the minor keys `lhs` and `rhs`. 
    /// 
    fn judge_lt(
        &self, 
        lhs: &SimplexFiltered<Filtration>, 
        rhs: &SimplexFiltered<Filtration>
    ) -> bool

    {
        let lhs_birth = self.relative_cycle_birth(lhs); 
        let rhs_birth = self.relative_cycle_birth(rhs); 
        // CASE 1: both chains are born as a relative cycle 
        if lhs_birth.is_some() && rhs_birth.is_some() { 
            // CASE 1.1: relative cycles born at the same time 
            if lhs_birth.unwrap() == rhs_birth.unwrap() { 
                return lhs.vertices.cmp(&rhs.vertices) == Ordering::Less;  
            }
            // CASE 1.2: relative cycles born at different time 
            else { 
                return lhs_birth.unwrap() < rhs_birth.unwrap();
            }
        }
        // CASE 2: lhs is a relative cycle and rhs is not relative cycle
        else if lhs_birth.is_some() && rhs_birth.is_none() {
            true
        }
        // CASE 3: lhs is not a relative cycle and rhs is a relative cycle
        else if lhs_birth.is_none() && rhs_birth.is_some() { 
            false
        }
        // CASE 4: lhs and rhs are not relative cycles
        else { 
            return lhs.vertices.cmp(&rhs.vertices) == Ordering::Less; 
        }
    }
}

/// Alternative implementation of JudgePartialOrder for `SourceColumnIndexOrderOperator`
/// This implementation operates on entries (not keys) of a boundary oracle by comparing the keys of each entry. 
/// This is useful for constructing major or minor views whose order are inherited from keys. 
impl<RelativeOracle, Filtration, Coefficient, RingOperator, OrderOperatorOracleKeyMinor, OrderOperatorOracleKeyMajor, OrderOperatorOracleViewMinor, OrderOperatorOracleViewMajor, OrderOperatorSubComplex>
    JudgePartialOrder<(SimplexFiltered<Filtration>, Coefficient)> for 
        SourceColumnIndexOrderOperator<RelativeOracle, Filtration, Coefficient, RingOperator, OrderOperatorOracleKeyMinor, OrderOperatorOracleKeyMajor, OrderOperatorOracleViewMinor, OrderOperatorOracleViewMajor, OrderOperatorSubComplex>
            where 
                RelativeOracle: Clone + VariableSortOracleKeys<SimplexFiltered<Filtration>, OrderOperatorOracleKeyMajor, OrderOperatorOracleKeyMinor, OrderOperatorSubComplex> + FactorFromArc<RelativeOracle, RingOperator, OrderOperatorOracleKeyMinor, OrderOperatorOracleKeyMajor>,             
                Arc<RelativeOracle>: ViewRowAscend + ViewColDescend + IndicesAndCoefficients<RowIndex=SimplexFiltered<Filtration>, ColIndex=SimplexFiltered<Filtration>, Coefficient=Coefficient, EntryMajor=(SimplexFiltered<Filtration>, Coefficient), EntryMinor=(SimplexFiltered<Filtration>, Coefficient)>, 
                <Arc<RelativeOracle> as IndicesAndCoefficients>::EntryMajor: Clone + KeyValSet<SimplexFiltered<Filtration>, Coefficient> + KeyValNew<SimplexFiltered<Filtration>, Coefficient> + KeyValGet<SimplexFiltered<Filtration>, Coefficient>, 
                <Arc<RelativeOracle> as IndicesAndCoefficients>::EntryMinor: Clone + KeyValSet<SimplexFiltered<Filtration>, Coefficient> + KeyValNew<SimplexFiltered<Filtration>, Coefficient> + KeyValGet<SimplexFiltered<Filtration>, Coefficient>,
                OrderOperatorOracleKeyMinor: JudgePartialOrder<SimplexFiltered<Filtration>> + SimplexDiameter<Filtration>, 
                OrderOperatorSubComplex: JudgePartialOrder<SimplexFiltered<Filtration>> + SimplexDiameter<Filtration>,        
                OrderOperatorOracleKeyMajor: JudgePartialOrder<SimplexFiltered<Filtration>>,                          
                OrderOperatorOracleViewMajor: Clone + JudgePartialOrder<(SimplexFiltered<Filtration>, Coefficient)>, 
                OrderOperatorOracleViewMinor: Clone + JudgePartialOrder<(SimplexFiltered<Filtration>, Coefficient)>,
                Filtration: Clone + Copy + Debug + PartialOrd + Ord + Hash,
                Coefficient: Clone + Debug + Hash,
                RingOperator: Semiring<Coefficient> + Ring<Coefficient> + DivisionRing<Coefficient> + Copy + Debug
{

    ///
    /// This function simply wraps a call to the existing implementation of `JudgePartialOrder` 
    /// for `SourceColumnIndexOrderOperator` which only judges oracle keys, or `SimplexFiltered` structs. 
    /// 
    fn judge_partial_cmp( 
        &self, 
        lhs: &(SimplexFiltered<Filtration>, Coefficient), 
        rhs: &(SimplexFiltered<Filtration>, Coefficient) 
    ) -> Option<Ordering> 
    
    {
        return self.judge_partial_cmp(&lhs.0, &rhs.0); 
    }
}

//  ===========================================================
//  Struct: TargetColumnIndexOrderOperator
//  ===========================================================

#[derive(Clone, Debug)]

/// Given the target (codomain) COMB for the U-Match of the boundary oracle of a filtered quotient, chain complex, this struct
/// provides methods for determining if/when a relative chain along the columns of this COMB is born as a relative boundary. 
/// Implements traits for placing a total order on the minor keys of this COMB according relative boundary birth of the 
/// corresponding minor views.  
/// 
/// Place a total order on the column indices (minor keys) of the target COMB matrix oracle of the U-Match decomposition of a 
/// `RelativeBoundaryMatrixOracle`. We refer to the U-Match simpy as TM = DS where D is a `RelativeBoundaryMatrixOracle` and T 
/// is the target COMB matrix oracle. The minor keys of the target COMB are the same simplices, or `SimplexFiltered` structs, 
/// which are the minor keys of the associated `RelativeBoundaryMatrixOracle`. Thus, this ordering on these simplices is not 
/// according to the birth of the indexing simplices themselves, but rather the birth of the chains along the columns they 
/// index as relative boundaries. 
/// 
/// This ordering on the columns of T allows for: 
/// 
/// (i) the construction of a matrix A which contains a filtered basis for relative boundaries. 
/// 
/// (ii) a second U-Match to be performed which "stitches" together one filtered basis containing all relative cycles and 
/// relative boundaries within its span.
/// 
/// The birth of the chain x as a relative boundary in the filtered quotient space K/K0 is given by min(a,b) where a is the 
/// birth of x in the sub complex filtration G and b is the birth of x as an absolute boundary in the full complex filtration F. 
/// 
/// Suppose that the ith column of T is indexed by the chain p and that likewise the jth column is indexed by the chain q. 
/// Let b(p) and b(q) denote their birth as relative boundaries computed as noted above. This order operator declares that the 
/// ith column of T appears before the jth column of S in the filtered basis A if and only if b(p) <= b(q). This ordering of 
/// columns prepares T for the second U-Match, which "stitches" together one basis containing all relative cycles and boundaries 
/// within its span. 
/// 
/// Any chain x never born as a realtive boundary will be assigned b(x) = INF. These chains are then ordered by lexicographically 
/// comparing the corresponding indexing simplices.  
/// 
pub struct TargetColumnIndexOrderOperator<
    RelativeOracle, Filtration, Coefficient, RingOperator, 
    OrderOperatorOracleKeyMinor, OrderOperatorOracleKeyMajor, 
    OrderOperatorOracleViewMinor, OrderOperatorOracleViewMajor, 
    OrderOperatorSubComplex>
        where 
            // trait bounds are identical to those of `RelativeBoundaryMatrixOracleWrapper`
            RelativeOracle: Clone + VariableSortOracleKeys<SimplexFiltered<Filtration>, OrderOperatorOracleKeyMajor, OrderOperatorOracleKeyMinor, OrderOperatorSubComplex> + FactorFromArc<RelativeOracle, RingOperator, OrderOperatorOracleKeyMinor, OrderOperatorOracleKeyMajor>,             
            Arc<RelativeOracle>: ViewRowAscend + ViewColDescend + IndicesAndCoefficients<RowIndex=SimplexFiltered<Filtration>, ColIndex=SimplexFiltered<Filtration>, Coefficient=Coefficient, EntryMajor=(SimplexFiltered<Filtration>, Coefficient), EntryMinor=(SimplexFiltered<Filtration>, Coefficient)>, 
            <Arc<RelativeOracle> as IndicesAndCoefficients>::EntryMajor: Clone + KeyValSet<SimplexFiltered<Filtration>, Coefficient> + KeyValNew<SimplexFiltered<Filtration>, Coefficient> + KeyValGet<SimplexFiltered<Filtration>, Coefficient>,
            <Arc<RelativeOracle> as IndicesAndCoefficients>::EntryMinor: Clone + KeyValSet<SimplexFiltered<Filtration>, Coefficient> + KeyValNew<SimplexFiltered<Filtration>, Coefficient> + KeyValGet<SimplexFiltered<Filtration>, Coefficient>,
            OrderOperatorOracleKeyMinor: JudgePartialOrder<SimplexFiltered<Filtration>> + SimplexDiameter<Filtration>,    
            OrderOperatorSubComplex: JudgePartialOrder<SimplexFiltered<Filtration>> + SimplexDiameter<Filtration>,        
            OrderOperatorOracleKeyMajor: JudgePartialOrder<SimplexFiltered<Filtration>>,                          
            OrderOperatorOracleViewMajor: Clone + JudgePartialOrder<(SimplexFiltered<Filtration>, Coefficient)>, 
            OrderOperatorOracleViewMinor: Clone + JudgePartialOrder<(SimplexFiltered<Filtration>, Coefficient)>,
            Filtration: Clone + Copy + Debug + PartialOrd + Ord + Hash,
            Coefficient: Clone + Debug + Hash,
            RingOperator: Semiring<Coefficient> + Ring<Coefficient> + DivisionRing<Coefficient> + Copy + Debug
{
    /// The U-Match deomposition of a `RelativeBoundaryMatrixOracle`. 
    relative_umatch: Umatch<Arc<RelativeOracle>, RingOperator, OrderOperatorOracleViewMajor, OrderOperatorOracleViewMinor>, 
    /// An order operator comparing simplices in the the filtration G on the sub complex K0. 
    order_operator_sub_complex_filtration: OrderOperatorSubComplex,           
    /// An order operator comparing simplices in the filtration F on the full complex K. 
    order_operator_full_complex_filtration: OrderOperatorOracleKeyMinor, 
    /// An order operator for the major keys of the factored `RelativeOracle`
    order_operator_oracle_key_major_phantom: PhantomData<OrderOperatorOracleKeyMajor>
}  

/// Implementation and methods for `TargetColumnIndexOrderOperator`
impl<RelativeOracle, Filtration, Coefficient, RingOperator, OrderOperatorOracleKeyMinor, OrderOperatorOracleKeyMajor, OrderOperatorOracleViewMinor, OrderOperatorOracleViewMajor, OrderOperatorSubComplex>
    TargetColumnIndexOrderOperator<RelativeOracle, Filtration, Coefficient, RingOperator, OrderOperatorOracleKeyMinor, OrderOperatorOracleKeyMajor, OrderOperatorOracleViewMinor, OrderOperatorOracleViewMajor, OrderOperatorSubComplex>
        where 
            RelativeOracle: Clone + VariableSortOracleKeys<SimplexFiltered<Filtration>, OrderOperatorOracleKeyMajor, OrderOperatorOracleKeyMinor, OrderOperatorSubComplex> + FactorFromArc<RelativeOracle, RingOperator, OrderOperatorOracleKeyMinor, OrderOperatorOracleKeyMajor>,             
            Arc<RelativeOracle>: ViewRowAscend + ViewColDescend + IndicesAndCoefficients<RowIndex=SimplexFiltered<Filtration>, ColIndex=SimplexFiltered<Filtration>, Coefficient=Coefficient, EntryMajor=(SimplexFiltered<Filtration>, Coefficient), EntryMinor=(SimplexFiltered<Filtration>, Coefficient)>, 
            <Arc<RelativeOracle> as IndicesAndCoefficients>::EntryMajor: Clone + KeyValSet<SimplexFiltered<Filtration>, Coefficient> + KeyValNew<SimplexFiltered<Filtration>, Coefficient> + KeyValGet<SimplexFiltered<Filtration>, Coefficient>, 
            <Arc<RelativeOracle> as IndicesAndCoefficients>::EntryMinor: Clone + KeyValSet<SimplexFiltered<Filtration>, Coefficient> + KeyValNew<SimplexFiltered<Filtration>, Coefficient> + KeyValGet<SimplexFiltered<Filtration>, Coefficient>,
            OrderOperatorOracleKeyMinor: JudgePartialOrder<SimplexFiltered<Filtration>> + SimplexDiameter<Filtration>,    
            OrderOperatorSubComplex: JudgePartialOrder<SimplexFiltered<Filtration>> + SimplexDiameter<Filtration>,        
            OrderOperatorOracleKeyMajor: JudgePartialOrder<SimplexFiltered<Filtration>>,                          
            OrderOperatorOracleViewMajor: Clone + JudgePartialOrder<(SimplexFiltered<Filtration>, Coefficient)>, 
            OrderOperatorOracleViewMinor: Clone + JudgePartialOrder<(SimplexFiltered<Filtration>, Coefficient)>,
            Filtration: Clone + Copy + Debug + PartialOrd + Ord + Hash,
            Coefficient: Clone + Debug + Hash,
            RingOperator: Semiring<Coefficient> + Ring<Coefficient> + DivisionRing<Coefficient> + Copy + Debug
{
    /// 
    /// Construct a `TargetColumnIndexOrderOperator` structure. 
    /// 
    /// User must provide the U-Match decomposition of a relative boundary matrix and 
    /// the order operators on the filtration of the full and sub complexs associated 
    /// with the relative boundary matrix. 
    /// 
    pub fn new(
        relative_umatch: Umatch<Arc<RelativeOracle>, RingOperator, OrderOperatorOracleViewMajor, OrderOperatorOracleViewMinor>,  
        order_operator_sub_complex_filtration: OrderOperatorSubComplex,           
        order_operator_full_complex_filtration: OrderOperatorOracleKeyMinor, 
    ) -> Self
    
    { 
        TargetColumnIndexOrderOperator { 
            relative_umatch, 
            order_operator_sub_complex_filtration, 
            order_operator_full_complex_filtration,
            order_operator_oracle_key_major_phantom: PhantomData
        }
    } 

    /// 
    /// Returns the filtration parameter where column T[:row_index] of the 
    /// target COMB is born as a relative boundary. Note that `row_index` refers
    /// to a simplex which is a row index (major key) of the associated 
    /// `RelativeBoundaryMatrixOracle` D and a column index (minor key) of T. 
    /// In the case that this column vector/chain is not a relative boundary, 
    /// then `None` is returned. 
    /// 
    pub fn relative_boundary_birth(
        &self, 
        row_index: &SimplexFiltered<Filtration>
    ) -> Option<Filtration>
        
    { 
        // STEP 1: Determine if the chain T[:,row_index] is born as an absolute boundary in the full complex filtration F
        // ======================================================================================================================================= 
        // - `matching_minor_key` is the minor key (column index) of M which is the location of the nonzero entry in the major view (row vector) 
        // M[row_index,:]
        // - note that `matching_minor_key` is `None` in the case that M[row_index,:] does not contain a nonzero entry
        // - Recall: Columns of T which correspond to nonzero rows of M constitute a basis for absolute boundaries of the full complex K
        // - Thus, the birth of the indexing simplex `matching_minor_key` is the birth of [T:,row_index] as an absolute boundary
        // - if `matching_minor_key` is `None`, then [T:,row_index] is never an absolute boundary in the filtration F on K 
        // - this filtration value is given by `_a` 
        // =======================================================================================================================================
        let matching_minor_key = self.relative_umatch.matching_ref().keymaj_to_keymin(&row_index); 
        let mut _a = None; 
        if matching_minor_key.is_some() { 
            _a = self.order_operator_full_complex_filtration.diameter(&matching_minor_key.as_ref().unwrap().vertices);
        }

        // STEP 2: Determine the birth of the chain given by T[:,row_index] in the subcomplex filtration G
        // =======================================================================================================================================
        // - `target_minor_view` is the column vector/chain given by T[:,row_index], which is just a linear combination of simplices
        // - FACT 1: T is upper triangular and, in OAT, the minor view T[:,row_index] will contain all major entries sorted by descending 
        // (bottom to top) order of index
        // - FACT 2: the major keys of T are identical to the major keys of D
        // - FACT 3: major keys of D (from bottom to top) are simplices in K ordered by F followed by simplces K \ K0 in ordered by G with 
        // filtration/diameter decreasing as row index decreases
        // - Given the above facts, then the first entry of `target_minor_view` will determine the birth of the chain T[:,row_index] in G: 
        //      (i.) if the first key of `target_minor_view` is a simplex x with G(x) = None, then T[:,row_index] is never born as a chain in G
        //      (ii.) if the first key of `target_minor_view` is a simplex x with G(x) being finite, then G(x) is the birth of T[:,row_index] as a 
        //      chain in G. Any other key y appears before x along the rows of D, meaning that G(y) is finite with G(y) <= G(x) by construction. 
        // - this filtration value is given by `_b`
        // =======================================================================================================================================
        let target_minor_view: Vec<(SimplexFiltered<Filtration>, Coefficient)> = self.relative_umatch.comb_codomain().view_minor_descend(row_index.clone()).collect();  
        let first_entry_major_key: SimplexFiltered<Filtration> = target_minor_view.iter().next().unwrap().key();
        let _b = self.order_operator_sub_complex_filtration.diameter(&first_entry_major_key.vertices);
        
        // STEP 3: Return the minimum of the filtration values `_a`` and `_b`
        // =======================================================================================================================================
        // - The birth of a chain as a relative boundary is the first time in which either of the following conditions hold:
        //      (i.) the chain has been born in the filtration G on K0
        //      (ii.) the chain has been born as an absolute boundary in the filtration F on K
        // - Equivalenlty, a basis for all relative boundaries is given by a basis for all absolute boundaries together with a basis for all chains 
        // in K0 
        // =======================================================================================================================================
        if _a.is_none() && _b.is_none() { 
            return None;
        }
        else if _a.is_some() && _b.is_none() { 
            return _a; 
        }
        else if _a.is_none() && _b.is_some() { 
            return _b;
        }
        else { 
            return Some(std::cmp::min(_a.unwrap(),_b.unwrap())); 
        }
    }

    ///
    /// Determine if a given chain, or column vector of T, is a 
    /// relative boundary. The user must provide the `SimplexFiltered` 
    /// struct indexing that column. 
    /// 
    /// This function simply wraps a call to `self.relative_boundary_birth()`. 
    /// 
    pub fn is_relative_boundary(
        &self, 
        row_index: &SimplexFiltered<Filtration>
    ) -> bool

    { 
        return self.relative_boundary_birth(row_index).is_some(); 
    }
}

/// Implement JudgePartialOrder for `TargetColumnIndexOrderOperator`
impl<RelativeOracle, Filtration, Coefficient, RingOperator, OrderOperatorOracleKeyMinor, OrderOperatorOracleKeyMajor, OrderOperatorOracleViewMinor, OrderOperatorOracleViewMajor, OrderOperatorSubComplex> 
    JudgePartialOrder<SimplexFiltered<Filtration>> for 
        TargetColumnIndexOrderOperator<RelativeOracle, Filtration, Coefficient, RingOperator, OrderOperatorOracleKeyMinor, OrderOperatorOracleKeyMajor, OrderOperatorOracleViewMinor, OrderOperatorOracleViewMajor, OrderOperatorSubComplex>
            where 
                RelativeOracle: Clone + VariableSortOracleKeys<SimplexFiltered<Filtration>, OrderOperatorOracleKeyMajor, OrderOperatorOracleKeyMinor, OrderOperatorSubComplex> + FactorFromArc<RelativeOracle, RingOperator, OrderOperatorOracleKeyMinor, OrderOperatorOracleKeyMajor>,             
                Arc<RelativeOracle>: ViewRowAscend + ViewColDescend + IndicesAndCoefficients<RowIndex=SimplexFiltered<Filtration>, ColIndex=SimplexFiltered<Filtration>, Coefficient=Coefficient, EntryMajor=(SimplexFiltered<Filtration>, Coefficient), EntryMinor=(SimplexFiltered<Filtration>, Coefficient)>, 
                <Arc<RelativeOracle> as IndicesAndCoefficients>::EntryMajor: Clone + KeyValSet<SimplexFiltered<Filtration>, Coefficient> + KeyValNew<SimplexFiltered<Filtration>, Coefficient> + KeyValGet<SimplexFiltered<Filtration>, Coefficient>, 
                <Arc<RelativeOracle> as IndicesAndCoefficients>::EntryMinor: Clone + KeyValSet<SimplexFiltered<Filtration>, Coefficient> + KeyValNew<SimplexFiltered<Filtration>, Coefficient> + KeyValGet<SimplexFiltered<Filtration>, Coefficient>,
                OrderOperatorOracleKeyMinor: JudgePartialOrder<SimplexFiltered<Filtration>> + SimplexDiameter<Filtration>,    
                OrderOperatorSubComplex: JudgePartialOrder<SimplexFiltered<Filtration>> + SimplexDiameter<Filtration>,        
                OrderOperatorOracleKeyMajor: JudgePartialOrder<SimplexFiltered<Filtration>>,                          
                OrderOperatorOracleViewMajor: Clone + JudgePartialOrder<(SimplexFiltered<Filtration>, Coefficient)>, 
                OrderOperatorOracleViewMinor: Clone + JudgePartialOrder<(SimplexFiltered<Filtration>, Coefficient)>,
                Filtration: Clone + Copy + Debug + PartialOrd + Ord + Hash,
                Coefficient: Clone + Debug + Hash,
                RingOperator: Semiring<Coefficient> + Ring<Coefficient> + DivisionRing<Coefficient> + Copy + Debug
{
    /// 
    /// Detemrine if a given chain `lhs` should appear before a chain `rhs` in 
    /// the filtration on relative boundaries along the columns of the target COMB. 
    /// If this is true, then `Some(Ordering::Less)` is returned. The user must 
    /// provide the `SimplexFiltered` structs which index the columns of T to 
    /// be compared. 
    /// 
    /// This function simply wraps a call to `self.judge_lt()`.
    /// 
    fn judge_partial_cmp(
        &self, 
        lhs: &SimplexFiltered<Filtration>, 
        rhs: &SimplexFiltered<Filtration>
    ) -> Option<Ordering>

    { 
        if self.judge_lt(lhs,rhs) { 
            return Some(Ordering::Less); 
        } else { 
            return Some(Ordering::Greater);
        }
    }

    ///
    /// Return true iff `lhs` < `rhs`. In words, the chain/column vector 
    /// corresponding to the provided minor key `lhs` should appear before the 
    /// chain/column vector corresponding to the provided minor key `rhs` in 
    /// the filtration of relative boundaries along the columns of the target COMB. 
    /// 
    /// Chains are first judged by birth as a relative boundaries using 
    /// `self.relative_boundary_birth()` and then by lexicographically comparing 
    /// indexing simplices (minor keys) in the case that either: 
    /// 
    /// (i.) two chains are born as relative boundaries at the same filtration/diameter
    /// 
    /// (ii.) two chains are both never relative boundaries, and have `None` as filtration 
    /// 
    /// The user must provide `SimplexFiltered` structs for the minor keys `lhs` and `rhs`. 
    /// 
    fn judge_lt(
        &self, 
        lhs: &SimplexFiltered<Filtration>, 
        rhs: &SimplexFiltered<Filtration>
    ) -> bool

    {
        let lhs_birth = self.relative_boundary_birth(lhs); 
        let rhs_birth = self.relative_boundary_birth(rhs); 
        // CASE 1: both chains are born as a relative boundary 
        if lhs_birth.is_some() && rhs_birth.is_some() { 
            // CASE 1.1 relative boundaries born at the same time 
            if lhs_birth.unwrap() == rhs_birth.unwrap() { 
                return lhs.vertices.cmp(&rhs.vertices) == Ordering::Less; 
            }
            // CASE 1.2 relative boundaries born at different time 
            else { 
                return lhs_birth.unwrap() < rhs_birth.unwrap() ;
            }
        }
        // CASE 2: lhs is a relative boundary and rhs is not relative boundary
        else if lhs_birth.is_some() && rhs_birth.is_none() {
            true
        }
        // CASE 3: lhs is not a relative boundary and rhs is a relative boundary
        else if lhs_birth.is_none() && rhs_birth.is_some() { 
            false
        }
        // CASE 4: lhs and rhs are not relative boundaries
        else { 
            return lhs.vertices.cmp(&rhs.vertices) == Ordering::Less; 
        }
    }
}

/// Alternative implementation of JudgePartialOrder for `TargetcolumnIndexORderOperator`
/// This implementation operates on entries (not keys) of a boundary oracle by comparing the keys of each entry. 
/// This is useful for constructing major or minor views whose order are inherited from keys.  
impl<RelativeOracle, Filtration, Coefficient, RingOperator, OrderOperatorOracleKeyMinor, OrderOperatorOracleKeyMajor, OrderOperatorOracleViewMinor, OrderOperatorOracleViewMajor, OrderOperatorSubComplex>
    JudgePartialOrder<(SimplexFiltered<Filtration>, Coefficient)> for 
        TargetColumnIndexOrderOperator<RelativeOracle, Filtration, Coefficient, RingOperator, OrderOperatorOracleKeyMinor, OrderOperatorOracleKeyMajor, OrderOperatorOracleViewMinor, OrderOperatorOracleViewMajor, OrderOperatorSubComplex>
            where 
                RelativeOracle: Clone + VariableSortOracleKeys<SimplexFiltered<Filtration>, OrderOperatorOracleKeyMajor, OrderOperatorOracleKeyMinor, OrderOperatorSubComplex> + FactorFromArc<RelativeOracle, RingOperator, OrderOperatorOracleKeyMinor, OrderOperatorOracleKeyMajor>,             
                Arc<RelativeOracle>: ViewRowAscend + ViewColDescend + IndicesAndCoefficients<RowIndex=SimplexFiltered<Filtration>, ColIndex=SimplexFiltered<Filtration>, Coefficient=Coefficient, EntryMajor=(SimplexFiltered<Filtration>, Coefficient), EntryMinor=(SimplexFiltered<Filtration>, Coefficient)>, 
                <Arc<RelativeOracle> as IndicesAndCoefficients>::EntryMajor: Clone + KeyValSet<SimplexFiltered<Filtration>, Coefficient> + KeyValNew<SimplexFiltered<Filtration>, Coefficient> + KeyValGet<SimplexFiltered<Filtration>, Coefficient>, 
                <Arc<RelativeOracle> as IndicesAndCoefficients>::EntryMinor: Clone + KeyValSet<SimplexFiltered<Filtration>, Coefficient> + KeyValNew<SimplexFiltered<Filtration>, Coefficient> + KeyValGet<SimplexFiltered<Filtration>, Coefficient>,
                OrderOperatorOracleKeyMinor: JudgePartialOrder<SimplexFiltered<Filtration>> + SimplexDiameter<Filtration>, 
                OrderOperatorSubComplex: JudgePartialOrder<SimplexFiltered<Filtration>> + SimplexDiameter<Filtration>,        
                OrderOperatorOracleKeyMajor: JudgePartialOrder<SimplexFiltered<Filtration>>,                          
                OrderOperatorOracleViewMajor: Clone + JudgePartialOrder<(SimplexFiltered<Filtration>, Coefficient)>, 
                OrderOperatorOracleViewMinor: Clone + JudgePartialOrder<(SimplexFiltered<Filtration>, Coefficient)>,
                Filtration: Clone + Copy + Debug + PartialOrd + Ord + Hash,
                Coefficient: Clone + Debug + Hash,
                RingOperator: Semiring<Coefficient> + Ring<Coefficient> + DivisionRing<Coefficient> + Copy + Debug
{

    ///
    /// This function simply wraps a call to the existing implementation of `JudgePartialOrder` 
    /// for `SourceColumnIndexOrderOperator` which only judges oracle keys, or `SimplexFiltered` structs. 
    /// 
    fn judge_partial_cmp( 
        &self, 
        lhs: &(SimplexFiltered<Filtration>, Coefficient), 
        rhs: &(SimplexFiltered<Filtration>, Coefficient) 
    ) -> Option<Ordering> 
    
    {
        return self.judge_partial_cmp(&lhs.0, &rhs.0); 
    }
}

//  ===========================================================
//  Struct: HashMapOrderOperator
//  ===========================================================

#[derive(Clone, Debug)]

///
/// An order operator introduced as a wrapper for custom order operators that: 
/// 
/// 1) are repeatedly used, and
/// 2) have a computationally expensive means of comparing items to sort.
/// 
/// The order operators [`SourceColumnIndexOrderOperator`] and [`TargetColumnIndexOrderOperator`] of this file
/// are the motivation for introducing this struct. 
/// 
/// Concisely, this order operator takes an unsorted vector holding structs of generic type `Item` and an `OrderOperator`. It 
/// sorts the list (only ONCE) using the `OrderOperator` and creates a bijective hash map `H: Item -> usize` between items of 
/// the unsorted vector and integer indices of the sorted vector. Thus, we have a < b iff H(a) < H(b), where the strict order 
/// only holds if the type `OrderOperator` defines a total order on the set `Vec<Item>`. 
/// 
/// Importantly, we note that:
/// 
/// 1) The hashing operation is optimized to be, on average, O(1) time. 
/// 2) Rust uses A SINGLE machine level operation to compare `usize` types ( such as the operation H(a) < H(b) ) which is 
/// ALWAYS ~1 CPU operation and O(1) time. 
/// 
pub struct HashMapOrderOperator<Item>
    where   
        Item: Clone + Eq + Hash

{   
    /// The sorted `Vec` of `Item`s
    pub sorted: Vec<Item>,
    /// A hash map `H: Item -> usize` taking an `Item` to its usize index in the vector `sorted`.
    pub hash_sorter: HashMap<Item, usize>
}  

/// Implementation and methods of `HashMapOrderOperator`
impl<Item> 
    HashMapOrderOperator<Item>
        where 
            Item: Clone + Eq + Hash 
{

    ///
    /// Construct an instance of `HashMapOrderOperator`
    /// 
    pub fn new<OrderOperator>(
        unsorted: Vec<Item>, 
        order_operator: OrderOperator
    ) -> Self

        where 
            OrderOperator: JudgePartialOrder<Item>

    {
        let mut sorted = unsorted.clone(); 
        sorted.sort_by(
            |lhs, rhs| order_operator.judge_partial_cmp(&lhs, &rhs).unwrap()
        );
        let h: HashMap<Item, usize> = sorted
            .iter()
            .enumerate()
            .map(|(i, t)| (t.clone(), i)) 
            .collect();
        HashMapOrderOperator { sorted: sorted, hash_sorter: h }
    }

}

/// Implement JudgePartialOrder for `HashMapOrderOperator`
impl<Item> 
    JudgePartialOrder<Item> for 
        HashMapOrderOperator<Item>
            where 
                Item: Clone + Eq + Hash
{
    /// 
    /// Use to comparison `self.judge_lt(lhs, rhs)` to determine a (total) ordering for `lhs` and `rhs`.
    /// 
    fn judge_partial_cmp(
        &self, 
        lhs: &Item, 
        rhs: &Item
    ) -> Option<Ordering>

    { 
        if self.judge_lt(lhs,rhs) { 
            return Some(Ordering::Less); 
        } else { 
            return Some(Ordering::Greater);
        }
    }

    ///
    /// Determine if `lhs` < `rhs` via the comparsion H(lhs) < H(rhs), where H is a hash map
    /// taking `lhs` and `rhs` from generic type `Item` to `usize`. 
    /// 
    /// We return `true` if `lhs` < `rhs`. 
    /// 
    fn judge_lt(
        &self, 
        lhs: &Item, 
        rhs: &Item
    ) -> bool

    {
        // the look-up is ~O(1) time on average and the < comparison is ALWAYS O(1) time! 
        if self.hash_sorter[&lhs] < self.hash_sorter[&rhs] { 
            return true; 
        } else { 
            return false;
        }
    }
}

/// Alternative implementation of JudgePartialOrder for `HashMapOrderOperator`
/// This implementation operates on tuples `T` such that `T.0` is of generic type `Item`
impl<Item, Other> 
    JudgePartialOrder<(Item, Other)> for 
        HashMapOrderOperator<Item>
            where 
                Item: Clone + Eq + Hash
{
    /// 
    /// Use to comparison `self.judge_lt(lhs, rhs)` to determine a (total) ordering for `lhs` and `rhs`.
    /// 
    fn judge_partial_cmp(
        &self, 
        lhs: &(Item, Other), 
        rhs: &(Item, Other)
    ) -> Option<Ordering>

    { 
        if self.judge_lt(lhs,rhs) { 
            return Some(Ordering::Less); 
        } else { 
            return Some(Ordering::Greater);
        }
    }

    ///
    /// Determine if `lhs` < `rhs` via the comparsion H(lhs) < H(rhs), where H is a hash map
    /// taking `lhs` and `rhs` from generic type `Item` to `usize`. 
    /// 
    /// We return `true` if `lhs` < `rhs`. 
    /// 
    fn judge_lt(
        &self, 
        lhs: &(Item, Other), 
        rhs: &(Item, Other)
    ) -> bool

    {
        // the look-up is ~O(1) time on average and the < comparison is ALWAYS O(1) time! 
        if self.hash_sorter[&lhs.0] < self.hash_sorter[&rhs.0] { 
            return true; 
        } else { 
            return false;
        }
    }
}