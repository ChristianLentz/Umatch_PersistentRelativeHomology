//! 
//! Traits of the oracle and order operators structs reuqired to compute persistent relative homology. 
//! 

use std::clone::Clone;
use std::fmt::Debug;
use std::hash::Hash;
use std::sync::Arc;
use core::option::Option;

use crate::algebra::vectors::entries::KeyValGet;
use crate::utilities::order::{JudgePartialOrder, OrderOperatorByKeyCutsom}; 
use crate::algebra::matrices::query::{IndicesAndCoefficients, ViewRowAscend};
use crate::algebra::matrices::operations::umatch::row_major::Umatch;

//  ===========================================================
//  Trait: VariableSortOracleKeys
//  ===========================================================

///
/// Get and sort the keys which index the rows and columns of a matrix 
/// oracle. This trait is ideally implemented for a square matrix whose 
/// major and minor keys are identical up to ordering, such as the  
/// boundary matrix of a filtered, quotient chain complex. This trait also 
/// includes functions which return references to the relevant order operators. 
///  
pub trait VariableSortOracleKeys<Key, OrderOperatorKeyMajor, OrderOperatorKeyMinor, OrderOperatorSubComplex> 
    where 
        Key: Clone,
        OrderOperatorKeyMajor: JudgePartialOrder<Key>, 
        OrderOperatorKeyMinor: JudgePartialOrder<Key>, 
        OrderOperatorSubComplex: JudgePartialOrder<Key>
{ 
    ///
    /// Sort the keys which index the rows or columns of a matrix oracle.  
    /// 
    fn get_sorted_keys_major_or_minor(&self, major: bool) -> Vec<Key>; 

    ///
    /// Get the list of keys which index the rows and columns of a matrix oracle. 
    /// 
    fn get_key_list(&self) -> Vec<Key>; 

    ///
    /// Return the order operator for major keys of the oracle. 
    /// 
    fn order_operator_key_major_ref(&self) -> &OrderOperatorKeyMajor; 

    ///
    /// Return the order operator for major keys of the oracle. 
    /// 
    fn order_operator_key_minor_ref(&self) -> &OrderOperatorKeyMinor; 

    ///
    /// Return the subspace order operator used to construct `OrderOperatorKeyMajor`. 
    /// 
    fn order_operator_sub_complex_ref(&self) -> &OrderOperatorSubComplex; 
} 

//  ===========================================================
//  Trait: FactorFromArc
//  ===========================================================

///
/// U-Match factor a matrix oracle from an Arc reference. 
/// 
/// Arc is optimized for safe reference sharing across multiple threads, 
/// and is also optimized for efficiently cloning large, immutable data. 
///
pub trait FactorFromArc<Mapping, RingOperator, OrderOperatorViewMajor, OrderOperatorViewMinor>
    where 
        Arc<Mapping>: ViewRowAscend + IndicesAndCoefficients,
        <Arc<Mapping> as IndicesAndCoefficients>::RowIndex: Clone + Hash + Eq,
        <Arc<Mapping> as IndicesAndCoefficients>::ColIndex: Clone + Hash + Eq, 
        <Arc<Mapping> as IndicesAndCoefficients>::EntryMajor: KeyValGet<<Arc<Mapping> as IndicesAndCoefficients>::ColIndex, <Arc<Mapping> as IndicesAndCoefficients>::Coefficient>,
        OrderOperatorViewMajor: JudgePartialOrder<<Arc<Mapping> as IndicesAndCoefficients>::ColIndex>, 
        OrderOperatorViewMinor: JudgePartialOrder<<Arc<Mapping> as IndicesAndCoefficients>::RowIndex>
{       
    ///
    /// U-Match factor a matrix oracle from an Arc reference. 
    /// 
    /// Arc is optimized for safe reference sharing across multiple threads, 
    /// and is also optimized for efficiently cloning large, immutable data. For 
    /// more on Arc see the documentation below: 
    /// 
    /// https://doc.rust-lang.org/std/sync/struct.Arc.html 
    /// 
    fn factor_from_arc(
        &self
    ) -> Umatch<
            Arc<Mapping>, 
            RingOperator, 
            OrderOperatorByKeyCutsom<
                <Arc<Mapping> as IndicesAndCoefficients>::ColIndex, 
                <Arc<Mapping> as IndicesAndCoefficients>::Coefficient, 
                <Arc<Mapping> as IndicesAndCoefficients>::EntryMajor, 
                OrderOperatorViewMajor>, 
            OrderOperatorByKeyCutsom<
                <Arc<Mapping> as IndicesAndCoefficients>::RowIndex, 
                <Arc<Mapping> as IndicesAndCoefficients>::Coefficient, 
                <Arc<Mapping> as IndicesAndCoefficients>::EntryMinor, 
                OrderOperatorViewMinor>
        >;
}

//  ===========================================================
//  Trait: SimplexDiameter
//  ===========================================================

///
/// Compute the diameter of a simplex given its vertices.
/// 
/// This trait provides additional flexibility/customization for placing an order on a list of simplices. 
/// 
pub trait SimplexDiameter<Filtration>
    where  
        Filtration: Copy + Debug + PartialOrd + Ord
{
    ///
    /// Compute the diameter of a simplex given its vertices. 
    /// 
    fn diameter(&self, vertices: &Vec<u16>) -> Option<Filtration>; 

    ///
    /// Return the diameter of the empty simplex. 
    /// 
    /// - i.e. a number that is less than or equal to each structural nonzero 
    /// entry of the associated dissimilarity matrix. 
    /// 
    /// OR 
    /// 
    /// - the smallest diameter/filtration that can be assigned to a simplex. 
    /// 
    fn dissimilarity_min(&self) -> Filtration; 
}