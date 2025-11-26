//! Matrix-matrix and matrix-vector multiplication.
//! 
//! 
//! - to multiply a matrix with a matrix, use 
//!   - [ProductMatrix::new](crate::algebra::matrices::operations::multiply::ProductMatrix)
// //!   - [ProductMatrixLazyMajorAscendUnsimplified::new](crate::algebra::matrices::operations::multiply::ProductMatrixLazyMajorAscendUnsimplified)
//! - to multiply a matrix with a vector, use either the convenient [VectorOperations](crate::algebra::vectors::operations::VectorOperations) trait methods
//!   - [multiply_matrix](crate::algebra::vectors::operations::VectorOperations::multiply_matrix)
//!   - [multiply_matrix_major_ascend](crate::algebra::vectors::operations::VectorOperations::multiply_matrix_major_ascend)
//!   - [multiply_matrix_minor_descend](crate::algebra::vectors::operations::VectorOperations::multiply_matrix_minor_descend)
//!     
//!     or the following
//!   - [vector_matrix_multiply_major_ascend_simplified](    crate::algebra::matrices::operations::multiply::vector_matrix_multiply_major_ascend_simplified)
//!   - [vector_matrix_multiply_major_ascend_unsimplified](  crate::algebra::matrices::operations::multiply::vector_matrix_multiply_major_ascend_unsimplified)
//!   - [vector_matrix_multiply_minor_descend_simplified](   crate::algebra::matrices::operations::multiply::vector_matrix_multiply_minor_descend_simplified)
//!   - [vector_matrix_multiply_minor_descend_unsimplified]( crate::algebra::matrices::operations::multiply::vector_matrix_multiply_minor_descend_unsimplified)



use std::{collections::HashMap, iter::FromIterator};

use crate::{algebra::{matrices::{query::{IndicesAndCoefficients, ViewColDescend, ViewRowAscend}, types::vec_of_vec::sorted::VecOfVec}, rings::operator_traits::Semiring, vectors::operations::{LinearCombinationSimplified, LinearCombinationUnsimplified, VectorOperations}}, utilities::{iterators::merge::hit::hit_merge_by_predicate, order::{JudgePartialOrder, ReverseOrder}}};
use crate::algebra::vectors::entries::{KeyValGet, KeyValSet};
use std::hash::Hash;



//  MATRIX - VECTOR MULTIPLICATION
//  ===========================================================================


//  MAJOR ASCEND
//  ---------------------------------------------------------------------------

/// Returns the product of a matrix with a vector.
/// 
/// The resulting iterator is a linear combination of major views.  It returns entries in ascending (but **non-strictly** ascending) order of index, provided that
/// [ViewRowAscend](crate::algebra::matrices::query::ViewRowAscend) returns entries in 
/// ascending order according to index.
/// 
/// **Note** the resulting iterator is not simplified -- it may return several entries with the same index.
/// 
/// # Examples
/// 
/// If `A` is a **row**-major representation of a matrix, and `v` is a sparse vector, then this function returns `vA`, a linear combination of **rows** of `A`.
/// If instead the representation is column-major, then this function returns `Av`.
/// 
/// ```
/// use oat_rust::algebra::matrices::types::vec_of_vec::sorted::VecOfVec;
/// use oat_rust::algebra::rings::operator_structs::field_prime_order::PrimeOrderFieldOperator;
/// use oat_rust::utilities::order::OrderOperatorByKey;
/// use oat_rust::algebra::matrices::operations::multiply::vector_matrix_multiply_major_ascend_unsimplified;
/// 
/// // a row-major matrix representation with usize major keys and isize minor keys
/// let matrix      =   VecOfVec::new(   vec![   vec![ (0isize, 1), (1isize, 6) ], 
///                                                    vec![ (0isize, 1), (1isize, 1) ],     ]     );
/// // the vector [1, 1]
/// let vector      =   vec![ (0usize, 1), (1usize, 1) ];
/// 
/// // the product
/// let product     =   vector_matrix_multiply_major_ascend_unsimplified(
///                         vector,
///                         & matrix,
///                         PrimeOrderFieldOperator::new(7), // the finite field of order 7
///                         OrderOperatorByKey::new(), // this compares tuples according to their lexicographic order
///                     );
/// // the product should return the following sequence of entries: (0,1), (0,1), (1,1), (1,6)   
/// itertools::assert_equal( product, vec![(0isize,1usize), (0,1), (1,1), (1,6)] )
/// ```
pub fn vector_matrix_multiply_major_ascend_unsimplified 
            < Matrix, RingOperator, SparseVecIter, OrderOperator> 
            ( 
                sparse_vec:         SparseVecIter,
                matrix:             Matrix, 
                ring_operator:      RingOperator,
                order_operator:   OrderOperator,
            ) 
            ->
            LinearCombinationUnsimplified
                < Matrix::ViewMajorAscendIntoIter, Matrix::ColIndex, Matrix::Coefficient, RingOperator, OrderOperator >            
    where 
        Matrix:                         ViewRowAscend + IndicesAndCoefficients,
        Matrix::ViewMajorAscend:        IntoIterator,
        Matrix::EntryMajor:   KeyValGet < Matrix::ColIndex, Matrix::Coefficient > + KeyValSet < Matrix::ColIndex, Matrix::Coefficient >,
        RingOperator:                   Clone + Semiring< Matrix::Coefficient >, // ring operators must typically implement Clone for operations such as simplification and dropping zeros
        Matrix::Coefficient:                 Clone, // Clone is required for HitMerge
        OrderOperator:                Clone + JudgePartialOrder<  Matrix::EntryMajor >, // order comparators must often implement Clone when one wishes to construct new, ordered objects
        SparseVecIter:                  IntoIterator,
        SparseVecIter::Item:            KeyValGet < Matrix::RowIndex, Matrix::Coefficient >,
{
    // LinearCombinationUnsimplified{
    //     linear_combination_unsimplified:
            hit_merge_by_predicate(
                sparse_vec
                        .into_iter()
                        .map(   |x| 
                                matrix
                                    .view_major_ascend(x.key() )
                                    .into_iter()
                                    .scale( x.val(), ring_operator.clone() )
                        ),
                    order_operator
            )        
    // }
}


/// Returns the product of a matrix with a vector; the result is a linear combination of the 
/// matrix's major views.
/// 
/// The resulting iterator is a simplified linear combination of major views.  It returns entries in **strictly** ascending order of index, provided that
/// [ViewRowAscend](crate::algebra::matrices::query::ViewRowAscend) returns entries in 
/// ascending order according to index.  In particular, no two consequtive indices have the same
/// index.
/// 
/// # Examples
/// 
/// If `A` is a **row**-major representation of a matrix, and `v` is a sparse vector, then this function returns `vA`, a linear combination of **rows** of `A`.
/// If instead the representation is column-major, then this function returns `Av`.
/// 
/// ```
/// use oat_rust::algebra::matrices::operations::multiply::vector_matrix_multiply_major_ascend_simplified;
/// use oat_rust::algebra::matrices::types::vec_of_vec::sorted::VecOfVec;
/// use oat_rust::algebra::rings::operator_structs::field_prime_order::PrimeOrderFieldOperator;
/// use oat_rust::utilities::order::OrderOperatorByKey;
/// 
/// // a row-major matrix representation with usize major keys and isize minor keys
/// let matrix      =   VecOfVec::new(   vec![   vec![ (0isize, 1), (1isize, 6) ], 
///                                                    vec![ (0isize, 1), (1isize, 1) ],     ]     );
/// // the vector [1, 1]
/// let vector      =   vec![ (0usize, 1), (1usize, 1) ];
/// 
/// // the product
/// let product     =   vector_matrix_multiply_major_ascend_simplified(
///                         vector,
///                         & matrix,
///                         PrimeOrderFieldOperator::new(7), // the finite field of order 7
///                         OrderOperatorByKey::new(), // this compares tuples according to their lexicographic order
///                     );
/// // the product should equal vector [2 0]        
/// itertools::assert_equal( product, vec![(0isize, 2)] )
/// ```
pub fn vector_matrix_multiply_major_ascend_simplified 
            < Matrix, RingOperator, SparseVecIter, OrderOperator> 
            ( 
                sparse_vec:         SparseVecIter,
                matrix:             Matrix, 
                ring_operator:      RingOperator,
                order_operator:   OrderOperator,
            ) 
            ->
            LinearCombinationSimplified
                < Matrix::ViewMajorAscendIntoIter, Matrix::ColIndex, Matrix::Coefficient, RingOperator, OrderOperator >            
    where 
        Matrix:                         ViewRowAscend + IndicesAndCoefficients,
        Matrix::ViewMajorAscend:        IntoIterator,
        Matrix::EntryMajor:   KeyValGet < Matrix::ColIndex, Matrix::Coefficient > + KeyValSet < Matrix::ColIndex, Matrix::Coefficient >,
        Matrix::Coefficient:                 Clone, // Clone is required for HitMerge
        Matrix::ColIndex:                 PartialEq,
        RingOperator:                   Clone + Semiring< Matrix::Coefficient >, // ring operators must typically implement Clone for operations such as simplification and dropping zeros        
        OrderOperator:                Clone + JudgePartialOrder<  Matrix::EntryMajor >, // order comparators must often implement Clone when one wishes to construct new, ordered objects
        SparseVecIter:                  IntoIterator,
        SparseVecIter::Item:            KeyValGet < Matrix::RowIndex, Matrix::Coefficient >,
{
    vector_matrix_multiply_major_ascend_unsimplified( 
            sparse_vec,
            matrix,
            ring_operator.clone(),
            order_operator,
        )
        // .linear_combination_unsimplified
        .simplify(ring_operator)
}        



//  MINOR DESCEND
//  ---------------------------------------------------------------------------


/// Returns the product of a matrix with a vector; the resulting iterator is an unsimplified linear combination of the matrix's minor views.
/// 
/// The resulting iterator returns entries in descending (but **non-strictly** descending) order of index, provided that
/// [ViewRowAscend](crate::algebra::matrices::query::ViewRowDescend) returns entries in 
/// descending order according to index.
/// 
/// **Note** the resulting iterator is not simplified -- it may return several entries with the same index.
/// 
/// # Examples
/// 
/// If `A` is a **row**-major representation of a matrix, and `v` is a sparse vector, then this function returns `A v`, a linear combination of **columns** of `A`.
/// If instead the representation is column-major, then this function returns `vA`.
/// 
/// ```
/// use oat_rust::algebra::matrices::operations::multiply::vector_matrix_multiply_minor_descend_unsimplified;
/// use oat_rust::algebra::matrices::types::vec_of_vec::sorted::VecOfVec;
/// use oat_rust::algebra::rings::operator_structs::field_prime_order::PrimeOrderFieldOperator;
/// use oat_rust::utilities::order::OrderOperatorAuto;
/// 
/// // a row-major matrix representation with usize major keys and isize minor keys
/// let matrix      =   VecOfVec::new(   vec![   vec![ (0isize, 1), (1isize, 6) ], 
///                                                    vec![ (0isize, 1), (1isize, 1) ],     ]     );
/// // the vector [1, 1]
/// let vector      =   vec![ (0isize, 1), (1isize, 1) ];
/// 
/// // the product
/// let mut product     =   vector_matrix_multiply_minor_descend_unsimplified(
///                         vector,
///                         & matrix,
///                         PrimeOrderFieldOperator::new(7), // the finite field of order 7
///                         OrderOperatorAuto, // this compares tuples according to their lexicographic order
///                     );
/// // the product should return the following entries in sequence: (1,1), (1,1), (0,1), (0,6)       
/// itertools::assert_equal( product, vec![(1,1), (1,1), (0,6), (0,1)] )
/// ```
pub fn vector_matrix_multiply_minor_descend_unsimplified 
            < Matrix, RowIndex, ColIndex, Coefficient, RingOperator, SparseVecIter, OrderOperator> 
            ( 
                sparse_vec:         SparseVecIter,
                matrix:             Matrix, 
                ring_operator:      RingOperator,
                order_operator:   OrderOperator,
            ) 
            ->
            LinearCombinationUnsimplified
                < Matrix::ViewMinorDescendIntoIter, RowIndex, Coefficient, RingOperator, ReverseOrder< OrderOperator > >            
    where 
        Matrix:                             ViewColDescend + IndicesAndCoefficients< ColIndex = ColIndex >,
        Matrix::ViewMinorDescendIntoIter:   Iterator,        
        < Matrix::ViewMinorDescendIntoIter as Iterator >::Item:     KeyValSet< RowIndex, Coefficient >,
        RingOperator:                       Clone + Semiring< Coefficient >, // ring operators must typically implement Clone for operations such as simplification and dropping zeros
        Coefficient:                             Clone, // Clone is required for HitMerge
        OrderOperator:                    Clone + JudgePartialOrder<  <Matrix::ViewMinorDescendIntoIter as Iterator>::Item >, // order comparators must often implement Clone when one wishes to construct new, ordered objects
        SparseVecIter:                      IntoIterator,
        SparseVecIter::Item:                KeyValGet < ColIndex, Coefficient >,
{
    // LinearCombinationUnsimplified{
    //     linear_combination_unsimplified:
            hit_merge_by_predicate(
                sparse_vec
                        .into_iter()
                        .map(   |x| 
                                matrix
                                    .view_minor_descend(x.key() )
                                    .into_iter()
                                    .scale( x.val(), ring_operator.clone() )
                        ),
                    ReverseOrder::new( order_operator )
            )        
    // }
}


/// Returns the product of a matrix with a vector; the result is a linear combination of the matrix's minor views.
/// 
/// The resulting iterator is a simplified linear combinatino of minor views.  It returns entries in **strictly** descending order of index, provided that
/// [ViewRowAscend](crate::algebra::matrices::query::ViewRowAscend) returns entries in 
/// ascending order according to index.  In particular, no two consequtive indices have the same
/// index.
/// 
/// # Examples
/// 
/// If `A` is a **row**-major representation of a matrix, and `v` is a sparse vector, then this function returns `A v`, a linear combination of **columns** of `A`.
/// If instead the representation is column-major, then this function returns `vA`.
/// 
/// ```
/// use oat_rust::algebra::matrices::operations::multiply::vector_matrix_multiply_minor_descend_simplified;
/// use oat_rust::algebra::matrices::types::vec_of_vec::sorted::VecOfVec;
/// use oat_rust::algebra::rings::operator_structs::field_prime_order::PrimeOrderFieldOperator;
/// use oat_rust::utilities::order::OrderOperatorByKey;
/// 
/// // a row-major matrix representation with usize major keys and isize minor keys
/// let matrix      =   VecOfVec::new(   vec![   vec![ (0isize, 1), (1isize, 6) ], 
///                                                    vec![ (0isize, 1), (1isize, 1) ],     ]     );
/// // the vector [1, 1]
/// let vector      =   vec![ (0isize, 1), (1isize, 1) ];
/// 
/// // the product
/// let product     =   vector_matrix_multiply_minor_descend_simplified(
///                         vector,
///                         & matrix,
///                         PrimeOrderFieldOperator::new(7), // the finite field of order 7
///                         OrderOperatorByKey::new(), // this compares order of entries according to their indices
///                     );
/// // the product should equal vector [0 1]        
/// itertools::assert_equal( product, vec![(1usize, 2)] )
/// ```
pub fn vector_matrix_multiply_minor_descend_simplified 
            < Matrix, RowIndex, ColIndex, Coefficient, RingOperator, SparseVecIter, OrderOperator> 
            ( 
                sparse_vec:         SparseVecIter,
                matrix:             Matrix, 
                ring_operator:      RingOperator,
                order_operator:   OrderOperator,
            ) 
            ->
            LinearCombinationSimplified
                < Matrix::ViewMinorDescendIntoIter, RowIndex, Coefficient, RingOperator, ReverseOrder< OrderOperator > >

    where 
        Matrix:                         ViewColDescend< ColIndex = ColIndex, RowIndex = RowIndex >,
        Matrix::ViewMinorDescend:       IntoIterator,
        Matrix::EntryMinor:  KeyValGet < RowIndex, Coefficient > + KeyValSet < RowIndex, Coefficient >,
        RingOperator:                   Clone + Semiring< Coefficient >, // ring operators must typically implement Clone for operations such as simplification and dropping zeros
        Coefficient:                         Clone, // Clone is required for HitMerge
        RowIndex:                         std::cmp::PartialEq, // required by the struct that simplifies vectors (it has to compare the indices of different entries)
        OrderOperator:                Clone + JudgePartialOrder<  < Matrix::ViewMinorDescend as IntoIterator >::Item >, // order comparators must often implement Clone when one wishes to construct new, ordered objects
        SparseVecIter:                  IntoIterator,
        SparseVecIter::Item:            KeyValGet < ColIndex, Coefficient >
{

    vector_matrix_multiply_minor_descend_unsimplified( 
            sparse_vec,
            matrix,
            ring_operator.clone(),
            order_operator,
        )
        // .linear_combination_unsimplified
        .simplify(ring_operator)
}    


























//  MATRIX - MATRIX MULTIPLICATION
//  ===========================================================================

/// Lazy product of two matrix oracles, represented by another matrix oracle
/// 
/// This struct represnts the product of two matrices.  It contains only four essential pieces of data:
/// * a reference to a matrix, `A`,
/// * a reference to another matrix, `B`,
/// * an object that specifies the coefficient ring over which both matrices are defined
/// * an object that specifies a total order on indices (it should be compatible with the order in which `A` and `B` return entries)
/// 
/// When the user requests a *row*, the lazy object first (i) scales 
/// some rows of `B` by an appropriate factor, then (ii) merges those rows into a single iterator, `J`.  Iterator `J` returns
/// entries in sorted order (ascending), but does *not* simplify terms: in particular, it does not drop zero terms or combine terms that 
/// have the same coefficient, then (iii) wraps `J` in a [`Simplify`](`crate::algebra::vectors::transfomations::Simplify`) struct that
/// *does* combine terms and drop zeros.
/// 
/// When the user requests a *column*, the lazy object first (i) scales 
/// some columns of `B` by an appropriate factor, then (ii) merges those columns into a single iterator, `J`.  Iterator `J` returns
/// entries in sorted order (ascending), but does *not* simplify terms: in particular, it does not drop zero terms or combine terms that 
/// have the same coefficient, then (iii) wraps `J` in a [`Simplify`](`crate::algebra::vectors::transfomations::Simplify`) struct that
/// *does* combine terms and drop zeros.
/// 
/// **Major dimension matters**. The major dimension is the dimension you 
/// actually get when you call `A.view_major_ascend(..)`.  The reason for this is the following pair of facts,
/// (both facts follow from the prodecure by which rows/columns are constructed, as described above).
/// * If `A` and `B` are row-major, then the lazy product `multiply_matrix_major_ascend(A, B, ..)` represents `A * B`
/// * If `A` and `B` are col-major, then the lazy product `multiply_matrix_major_ascend(A, B, ..)` represents `B * A`
/// 
/// 
/// **Example**
/// Suppose that `X` represents the product of `A` and `B` (within `X`, we refer to `A` and `B` as `matrix_1` and `matrix_2`,
/// respectively).  If we call `X.view_major_ascend( i )`, then the following happens
/// 1) Let `(j1, a1)`, .., `(jn, an)` be the structural nonzero entries in the `i`th major view of `A`.
/// 2) Let `Jk` denote the iterator obtained by scaling row `jk` of matrix `B` by a factor of `ak`.
/// 3) Let `J` be the iterator obtained by merging `J1, .., Jn` via [hit_merge_by_predicate].
/// 4) Let `K` be the iterator obtained by combining all entries with the same index into a single entry, and dropping this entry if it is zero.
/// Then `K` is the output returned by `X.view_major_ascend( i )`.
/// 
/// # Examples
/// 
/// ```
/// // Import the necessary traits and structs
/// // ---------------------------------------------------------------------------------------
/// 
/// use oat_rust::algebra::rings::operator_structs::ring_native::DivisionRingNative;
/// use oat_rust::algebra::matrices::types::vec_of_vec::sorted::VecOfVec;
/// use oat_rust::algebra::matrices::query::{ViewRowAscend};
/// use oat_rust::algebra::matrices::operations::multiply::ProductMatrix;
/// use oat_rust::algebra::vectors::entries::KeyValGet;
/// use oat_rust::utilities::order::OrderOperatorByLessThan;
/// use std::iter::FromIterator;
/// 
/// // Initialize variables
/// // ---------------------------------------------------------------------------------------
/// 
/// // Define the operator for the coefficient ring
/// let ring_operator = DivisionRingNative::<f64>::new();     
/// // Define matrix A, a 2x2 matrix with 1's above the diagonal and 0's below
/// let matrix_1   =   VecOfVec::new(    vec![ vec![ (0,1.), (1,1.) ], vec![ (1,1.) ] ]    );
/// // Define matrix B, a 2x2 matrix with 1's above the diagonal and 0's below
/// let matrix_2   =   VecOfVec::new(    vec![ vec![ (0,1.), (1,1.) ], vec![ (1,1.) ] ]    );
/// // Define references to the two arrays (recall that ViewRowAscend is only implemented on `& VecOfVec`, not on `VecOfVec`)
/// let matrix_1_ref   =   &matrix_1;
/// let matrix_2_ref    =   &matrix_2;   
///      
/// // Define the lazy product of A and B
/// // ---------------------------------------------------------------------------------------
/// 
/// let product     =   ProductMatrix::new( 
///                             matrix_1_ref,                 // matrix A
///                             matrix_2_ref,                 // matrix B
///                             ring_operator.clone(),      // ring operator
///                             OrderOperatorByLessThan::new( |x:&(i32, f64), y:&(i32, f64)| x.key() < y.key() )   // defines a total order on indices
///                         );
///                     
/// // Check the output iterators themselves (these are wrapped in `Simplify` structs).
/// // ---------------------------------------------------------------------------------------
///                     
/// // Use this object to create a vector-of-vectors that represents the product
/// let output =   Vec::from_iter(
///                                         (0..2)
///                                             .map(   |x| 
///                                                     Vec::from_iter(
///                                                             product
///                                                             .view_major_ascend(x)
///                                                         )
///                                                 )
///                                     );
///                                 
/// // Check that the answer is correct                                    
/// assert_eq!(     
///         output,
///         vec![ vec![(0,1.), (1,2.)], vec![(1,1.)]  ]
///     ); 
/// 
/// // Check the underlying, unsimplified iterators (these are are contained inside the `Simplify` structs).
/// // -------------------------------------------------------------------------------------------------------------
/// 
/// // Use this object to create a vector-of-vectors that represents the product
/// let output =   Vec::from_iter(
///                                         (0..2)
///                                             .map(   |x| 
///                                                     Vec::from_iter(
///                                                             product
///                                                             .view_major_ascend(x)
///                                                             .unsimplified // this unwraps the inner iterator
///                                                         )
///                                                 )
///                                     );
///                                 
/// // Check that the answer is correct                                    
/// assert_eq!(     
///         output,
///         vec![ vec![(0,1.), (1,1.), (1,1.)], vec![(1,1.)]  ]
///     ); 
/// ```
/// 
/// # Limitations
/// 
/// This struct cannot return descending minor views of the product because it contains no order operator for the entries of minor views.
#[derive(Clone, Copy, Debug)]
pub struct ProductMatrix < 
                    Matrix1, 
                    Matrix2,                
                    RingOperator,
                    OrderOperator,
                > 
            where 
                Matrix1:                            ViewRowAscend + IndicesAndCoefficients,
                Matrix2:                            ViewRowAscend + IndicesAndCoefficients< Coefficient = Matrix1::Coefficient, RowIndex = Matrix1::ColIndex >, 
                Matrix1::ViewMajorAscend:           IntoIterator,
                Matrix2::ViewMajorAscend:           IntoIterator,
                Matrix1::EntryMajor:                KeyValGet < Matrix1::ColIndex, Matrix1::Coefficient >,
                Matrix2::EntryMajor:                KeyValGet < Matrix2::ColIndex, Matrix2::Coefficient >,  
                Matrix2::ColIndex:                  Clone,
                Matrix2::Coefficient:               Clone,                          
                RingOperator:                       Clone + Semiring< Matrix1::Coefficient >,
                OrderOperator:                      Clone + JudgePartialOrder<  Matrix2::EntryMajor >,                                                           
{
    matrix_1:                   Matrix1,     
    matrix_2:                   Matrix2,
    ring_operator:              RingOperator,
    order_operator:             OrderOperator,          
}

impl    < 
            Matrix1, 
            Matrix2,                
            RingOperator,
            OrderOperator,
        > 
    ProductMatrix <
            Matrix1, 
            Matrix2,                
            RingOperator,
            OrderOperator,
        > 
    where 
        Matrix1:                            ViewRowAscend + IndicesAndCoefficients,
        Matrix2:                            ViewRowAscend + IndicesAndCoefficients< Coefficient = Matrix1::Coefficient, RowIndex = Matrix1::ColIndex >, 
        Matrix1::ViewMajorAscend:           IntoIterator,
        Matrix2::ViewMajorAscend:           IntoIterator,
        Matrix1::EntryMajor:                KeyValGet < Matrix1::ColIndex, Matrix1::Coefficient >,
        Matrix2::EntryMajor:                KeyValGet < Matrix2::ColIndex, Matrix2::Coefficient >,  
        Matrix2::ColIndex:                  Clone,
        Matrix2::Coefficient:               Clone,                          
        RingOperator:                       Clone + Semiring< Matrix1::Coefficient >,
        OrderOperator:                      Clone + JudgePartialOrder<  Matrix2::EntryMajor >,    

{
    
    /// Generate a lazy lazy product of two matrix oracles
    /// 
    /// See documentation for [`ProductMatrix`].  In that discussion, the function arguments `matrix_1` and `matrix_2`
    /// correspond to matrices `A` and `B`, respectively.
    pub fn new( 
                    matrix_1:           Matrix1, 
                    matrix_2:           Matrix2, 
                    ring_operator:      RingOperator, 
                    order_operator:     OrderOperator 
                ) 
            -> Self {
        ProductMatrix{
            matrix_1,
            matrix_2,
            ring_operator,
            order_operator,         
        }
    }
  
}

// IndicesAndCoefficients
impl     < Matrix1, Matrix2, RingOperator, OrderOperator, > 

    IndicesAndCoefficients for

    ProductMatrix< Matrix1, Matrix2, RingOperator, OrderOperator, > 

    where
        Matrix1:                            ViewRowAscend + IndicesAndCoefficients,
        Matrix2:                            ViewRowAscend + IndicesAndCoefficients< Coefficient = Matrix1::Coefficient, RowIndex = Matrix1::ColIndex >, 
        Matrix1::ViewMajorAscend:           IntoIterator,
        Matrix2::ViewMajorAscend:           IntoIterator,
        Matrix1::EntryMajor:      KeyValGet < Matrix1::ColIndex, Matrix1::Coefficient >,
        Matrix2::EntryMajor:      KeyValGet < Matrix2::ColIndex, Matrix2::Coefficient >,  
        Matrix2::ColIndex:                    Clone,
        Matrix2::Coefficient:                    Clone,                          
        RingOperator:                       Clone + Semiring< Matrix1::Coefficient >,
        OrderOperator:                    Clone + JudgePartialOrder<  Matrix2::EntryMajor >,         
{
    type EntryMajor = Matrix2::EntryMajor;
    type EntryMinor = Matrix1::EntryMinor;
    type ColIndex = Matrix2::ColIndex; 
    type RowIndex = Matrix1::RowIndex; 
    type Coefficient = Matrix1::Coefficient;
}   

// ViewRowAscend
impl     < 
        Matrix1, 
        Matrix2, 
        RingOperator,
        OrderOperator,
        > 

    ViewRowAscend for
    
    ProductMatrix<
            Matrix1, 
            Matrix2, 
            RingOperator,
            OrderOperator,
        >     
    where
        Matrix1:                            ViewRowAscend + IndicesAndCoefficients,
        Matrix2:                            ViewRowAscend + IndicesAndCoefficients< Coefficient = Matrix1::Coefficient, RowIndex = Matrix1::ColIndex >, 
        Matrix1::ViewMajorAscend:           IntoIterator,
        Matrix2::ViewMajorAscend:           IntoIterator,
        Matrix1::EntryMajor:      KeyValGet < Matrix1::ColIndex, Matrix1::Coefficient >,
        Matrix2::EntryMajor:      KeyValGet < Matrix2::ColIndex, Matrix2::Coefficient > + KeyValSet < Matrix2::ColIndex, Matrix2::Coefficient >,  
        Matrix2::ColIndex:                    Clone + PartialEq, // PartialEq is required by the struct that simplifies sparse vector iterators; it has to be able to compare the indices of different entries
        Matrix2::Coefficient:                    Clone,                          
        RingOperator:                       Clone + Semiring< Matrix1::Coefficient >,
        OrderOperator:                    Clone + JudgePartialOrder<  Matrix2::EntryMajor >,                 

{   
    type ViewMajorAscend            =   LinearCombinationSimplified
                                            < Matrix2::ViewMajorAscendIntoIter, Matrix2::ColIndex, Matrix2::Coefficient, RingOperator, OrderOperator >;
    type ViewMajorAscendIntoIter    =   Self::ViewMajorAscend;

    fn view_major_ascend( & self, index: Self::RowIndex ) 
        -> 
        LinearCombinationSimplified< Matrix2::ViewMajorAscendIntoIter, Matrix2::ColIndex, Matrix2::Coefficient, RingOperator, OrderOperator >
    {

        vector_matrix_multiply_major_ascend_simplified( 
                self.matrix_1.view_major_ascend( index ),
                & self.matrix_2,
                self.ring_operator.clone(),
                self.order_operator.clone(),
            )   

    }
}



/// 
/// A lazy product of two matrices, capable of ascending row views and descending column views, each with a (possibly) unique order operator.  
/// 
/// This struct is designed similarly to the `ProductMatrix` contained in this module. It is recommended that the user 
/// refer to its documentation. This struct was introduced given a need for product oracles with descending column views, 
/// particularly for the module `oat_rust::src::algebra::chains::relative.rs`.
/// 
/// There are no unit tests included for this struct, as it is identical to the `ProductMatrix`.
/// 
#[derive(Debug, Clone)]
pub struct BimajorProductMatrix<LeftMatrix, RightMatrix, RingOperator, OrderOperatorKeyMajor, OrderOperatorViewMajor>
        where 
            // traits bounds of left matrix
            LeftMatrix: ViewRowAscend + IndicesAndCoefficients, 
            LeftMatrix::ViewMajorAscend: IntoIterator,
            LeftMatrix::EntryMajor: KeyValGet<LeftMatrix::ColIndex, LeftMatrix::Coefficient>,
            // trait bounds of right matrix 
            RightMatrix: ViewRowAscend + IndicesAndCoefficients<Coefficient=LeftMatrix::Coefficient, RowIndex=LeftMatrix::ColIndex>,  
            RightMatrix::ViewMajorAscend: IntoIterator, 
            RightMatrix::EntryMajor: KeyValGet<RightMatrix::ColIndex, RightMatrix::Coefficient>,
            RightMatrix::ColIndex: Clone, 
            RightMatrix::Coefficient: Clone, 
            // other trait bounds 
            RingOperator: Clone + Semiring<LeftMatrix::Coefficient>,
            // order operators 
            OrderOperatorViewMajor: JudgePartialOrder<RightMatrix::EntryMajor> + Clone, 
            OrderOperatorKeyMajor: JudgePartialOrder<LeftMatrix::RowIndex> + JudgePartialOrder<LeftMatrix::EntryMinor> + Clone
{   
    /// The left matrix of the filtered product. 
    pub left: LeftMatrix, 
    /// The right matrix of the filtered product
    pub right: RightMatrix,
    /// A ring operator for constructing major and minor views. 
    pub ring_operator: RingOperator, 
    /// A vector of major keys of the `BimajorProductMatrix`. This field is included so that the product oracle can be UMatch factored easily. 
    pub sorted_major_keys_of_product: Vec<LeftMatrix::RowIndex>, 
    /// An order operator for major keys of the `BimajorProductMatrix`. 
    pub order_operator_key_major: OrderOperatorKeyMajor, 
    /// An order operator for major entries of the `BimajorProductMatrix`. 
    pub order_operator_view_major: OrderOperatorViewMajor
}

/// Implementation and methods of `BimajorProductMatrix`
impl<'a, LeftMatrix, RightMatrix, RingOperator, OrderOperatorKeyMajor, OrderOperatorViewMajor>
    BimajorProductMatrix<LeftMatrix, RightMatrix, RingOperator, OrderOperatorKeyMajor, OrderOperatorViewMajor>
        where
            LeftMatrix: ViewRowAscend + IndicesAndCoefficients,  
            LeftMatrix::ViewMajorAscend: IntoIterator,
            LeftMatrix::EntryMajor: KeyValGet<LeftMatrix::ColIndex, LeftMatrix::Coefficient>,
            RightMatrix: ViewRowAscend + IndicesAndCoefficients<Coefficient=LeftMatrix::Coefficient, RowIndex=LeftMatrix::ColIndex>,  
            RightMatrix::ViewMajorAscend: IntoIterator, 
            RightMatrix::EntryMajor: KeyValGet<RightMatrix::ColIndex, RightMatrix::Coefficient>,
            RightMatrix::ColIndex: Clone, 
            RightMatrix::Coefficient: Clone,
            RingOperator: Clone + Semiring<LeftMatrix::Coefficient>,
            OrderOperatorViewMajor: JudgePartialOrder<RightMatrix::EntryMajor> + Clone, 
            OrderOperatorKeyMajor: JudgePartialOrder<LeftMatrix::RowIndex> + JudgePartialOrder<LeftMatrix::EntryMinor> + Clone
{ 
    /// 
    /// Construct a new `BimajorProductMatrix`. The user must provide 
    /// 
    /// - A `Left` and `Right` matrix to multiply. 
    /// - A `RingOperator` for the oracle. 
    /// - A list `major_keys_of_product` giving the row indices of the wrapped product oracle. The oracle owns a copy of this list 
    /// as this information is required to construct a U-Match. It is assumed that the provided list is sorted. 
    /// - Order operators for iterating over row indices keys constructing ascending row views of the product oracle, given by 
    /// `order_operator_key_major_of_product` and `order_operator_view_major_of_product` respectively.
    /// 
    pub fn new(
        left: LeftMatrix, 
        right: RightMatrix, 
        ring_operator: RingOperator,
        major_keys_of_product: Vec<LeftMatrix::RowIndex>, 
        order_operator_key_major_of_product: OrderOperatorKeyMajor, 
        order_operator_view_major_of_product: OrderOperatorViewMajor
    ) -> Self 
    
    { 
        // assert that the provided keys are sorted
        let is_sorted = major_keys_of_product.is_sorted_by(
            |lhs: &LeftMatrix::RowIndex, rhs: &LeftMatrix::RowIndex| order_operator_key_major_of_product.judge_partial_cmp(lhs,rhs).unwrap().is_le()
        );
        if !is_sorted { 
            panic!("\n\nError: Constructing `BimajorProductMatrix` failed. The list of row indices provided to the constructor is NOT sorted by the accompanying order operator on row indices. \n This message is generated by OAT.\n\n");
        }
        BimajorProductMatrix{ 
            left, 
            right, 
            ring_operator,
            sorted_major_keys_of_product: major_keys_of_product, 
            order_operator_key_major: order_operator_key_major_of_product, 
            order_operator_view_major: order_operator_view_major_of_product
        }
    }
}

/// Implement `IndicesAndCoefficients` for `BimajorProductMatrix`
impl<'a, LeftMatrix, RightMatrix, RingOperator, OrderOperatorKeyMajor, OrderOperatorViewMajor>
    IndicesAndCoefficients for 
        BimajorProductMatrix<LeftMatrix, RightMatrix, RingOperator, OrderOperatorKeyMajor, OrderOperatorViewMajor>
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
        BimajorProductMatrix<LeftMatrix, RightMatrix, RingOperator, OrderOperatorKeyMajor, OrderOperatorViewMajor>
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
    type ViewMajorAscend = LinearCombinationSimplified<RightMatrix::ViewMajorAscendIntoIter, RightMatrix::ColIndex, RightMatrix::Coefficient, RingOperator, OrderOperatorViewMajor>;
    type ViewMajorAscendIntoIter = Self::ViewMajorAscend;
    ///
    /// Obtain a major view of the `BimajorProductMatrix`. Returns a simplified linear combination of the rows of 
    /// `self.right`. This linear combination uses scalars from a row of `self.left`. The major entries of the returned 
    /// vector are sorted by an instance of local type `OrderOperatorViewMajor`. 
    /// 
    fn view_major_ascend(&self, index: Self::RowIndex) -> Self::ViewMajorAscend {
        let view_major_left: Vec<LeftMatrix::EntryMajor> = self.left.view_major_ascend(index).into_iter().collect();  
        vector_matrix_multiply_major_ascend_simplified( 
            view_major_left,
            &self.right,
            self.ring_operator.clone(),
            self.order_operator_view_major.clone(),
        )
    }
}

/// Implement `ViewColDescend` for `BimajorProductMatrix` 
impl<'a, LeftMatrix, RightMatrix, RingOperator, OrderOperatorKeyMajor, OrderOperatorViewMajor>
    ViewColDescend for 
        BimajorProductMatrix<LeftMatrix, RightMatrix, RingOperator, OrderOperatorKeyMajor, OrderOperatorViewMajor>
            where 
                LeftMatrix: ViewColDescend + ViewRowAscend + IndicesAndCoefficients,  
                LeftMatrix::ViewMinorDescend: IntoIterator,
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
    type ViewMinorDescend = LinearCombinationSimplified<LeftMatrix::ViewMinorDescendIntoIter, LeftMatrix::RowIndex, LeftMatrix::Coefficient, RingOperator, ReverseOrder<OrderOperatorKeyMajor>>;
    type ViewMinorDescendIntoIter = Self::ViewMinorDescend;
    ///
    /// Obtain a minor view of the `FilteredProductMatrix`. Returns a simplified linear combination of the columns of 
    /// `self.left`. This linear combination uses scalars from a column of `self.right`. The minor entries of the returned 
    /// vector are sorted by an instance of local type `OrderOperatorKeyMajor`. 
    /// 
    fn view_minor_descend(&self, index: Self::ColIndex) -> Self::ViewMinorDescend {  
        let view_minor_right: Vec<RightMatrix::EntryMinor> = self.right.view_minor_descend(index).into_iter().collect(); 
        vector_matrix_multiply_minor_descend_simplified( 
            view_minor_right,
            &self.left,
            self.ring_operator.clone(),
            self.order_operator_key_major.clone(),
        )
    }
}

///
/// A sparse representation of a [`BimajorProductMatrix`]. Given the lazy representtion of the oracle, we explicilty store the 
/// CSR and CSC product and use hash maps between generic indices and integers to generate ascending row views and descending column views. 
/// 
pub struct BimajorSparseProductMatrix<LeftMatrix, RightMatrix, RingOperator, OrderOperatorKeyMajor, OrderOperatorViewMajor>
   where 
        // traits bounds of left matrix
        LeftMatrix: ViewColDescend + ViewRowAscend + IndicesAndCoefficients, 
        LeftMatrix::ViewMajorAscend: IntoIterator,
        LeftMatrix::EntryMajor: KeyValGet<LeftMatrix::ColIndex, LeftMatrix::Coefficient>,
        LeftMatrix::EntryMinor: KeyValGet<LeftMatrix::RowIndex, LeftMatrix::Coefficient> + KeyValSet<LeftMatrix::RowIndex, LeftMatrix::Coefficient>,
        LeftMatrix::RowIndex: Eq + Hash + Clone,
        // trait bounds of right matrix 
        RightMatrix: ViewColDescend + ViewRowAscend + IndicesAndCoefficients<Coefficient=LeftMatrix::Coefficient, RowIndex=LeftMatrix::ColIndex>,  
        RightMatrix::ViewMajorAscend: IntoIterator, 
        RightMatrix::EntryMajor: KeyValGet<RightMatrix::ColIndex, RightMatrix::Coefficient> + KeyValSet<RightMatrix::ColIndex, RightMatrix::Coefficient>,
        RightMatrix::EntryMinor: KeyValGet<RightMatrix::RowIndex, RightMatrix::Coefficient>,
        RightMatrix::ColIndex: Eq + Hash + Clone, 
        RightMatrix::Coefficient: Clone, 
        // other trait bounds 
        RingOperator: Clone + Semiring<LeftMatrix::Coefficient>,
        // order operators 
        OrderOperatorViewMajor: JudgePartialOrder<RightMatrix::ColIndex> + JudgePartialOrder<RightMatrix::EntryMajor> + Clone, 
        OrderOperatorKeyMajor: JudgePartialOrder<LeftMatrix::RowIndex> + JudgePartialOrder<LeftMatrix::EntryMinor> + Clone
{   
    /// A CSR representtion of the product. 
    sorted_csr_ascending: Vec<Vec< (RightMatrix::ColIndex, LeftMatrix::Coefficient) >>,
    // A CSC representation of the product
    sorted_csc_descending: Vec<Vec< (LeftMatrix::RowIndex, LeftMatrix::Coefficient) >>, 
    /// Sorted row indices in ascending order. 
    pub sorted_row_indices: Vec<LeftMatrix::RowIndex>, 
    /// Sorted column indices in *ascending* order. 
    pub sorted_column_indices: Vec<RightMatrix::ColIndex>,
    /// A has map taking a sorted row index to a unique integer. 
    row_index_to_integer_bijection: HashMap<LeftMatrix::RowIndex, usize>, 
    /// A hash map taking a sorted column index to a unqie integer. 
    column_index_to_integer_bijection: HashMap<RightMatrix::ColIndex, usize>,
    /// The lazy product matrix from which we produce the compressed representaitons. 
    /// Includes references to the `Left` and `Right` matrices and their order operators. 
    pub lazy_product: BimajorProductMatrix<LeftMatrix, RightMatrix, RingOperator, OrderOperatorKeyMajor, OrderOperatorViewMajor>,
}

/// Implementation and methods for `BimajorSparseProductMatrix`
impl<LeftMatrix, RightMatrix, RingOperator, OrderOperatorKeyMajor, OrderOperatorViewMajor>
    BimajorSparseProductMatrix<LeftMatrix, RightMatrix, RingOperator, OrderOperatorKeyMajor, OrderOperatorViewMajor>
        where
            LeftMatrix: ViewColDescend + ViewRowAscend + IndicesAndCoefficients,  
            LeftMatrix::ViewMajorAscend: IntoIterator,
            LeftMatrix::EntryMajor: KeyValGet<LeftMatrix::ColIndex, LeftMatrix::Coefficient>,
            LeftMatrix::EntryMinor: KeyValGet<LeftMatrix::RowIndex, LeftMatrix::Coefficient> + KeyValSet<LeftMatrix::RowIndex, LeftMatrix::Coefficient>,
            LeftMatrix::RowIndex: Eq + Hash + Clone,
            RightMatrix: ViewColDescend + ViewRowAscend + IndicesAndCoefficients<Coefficient=LeftMatrix::Coefficient, RowIndex=LeftMatrix::ColIndex>,  
            RightMatrix::ViewMajorAscend: IntoIterator, 
            RightMatrix::EntryMajor: KeyValGet<RightMatrix::ColIndex, RightMatrix::Coefficient> + KeyValSet<RightMatrix::ColIndex, RightMatrix::Coefficient>,
            RightMatrix::EntryMinor: KeyValGet<RightMatrix::RowIndex, RightMatrix::Coefficient>,
            RightMatrix::ColIndex: Eq + Hash + Clone, 
            RightMatrix::Coefficient: Clone,
            RingOperator: Clone + Semiring<LeftMatrix::Coefficient>,
            OrderOperatorViewMajor: JudgePartialOrder<RightMatrix::ColIndex> + JudgePartialOrder<RightMatrix::EntryMajor> + Clone, 
            OrderOperatorKeyMajor: JudgePartialOrder<LeftMatrix::RowIndex> + JudgePartialOrder<LeftMatrix::EntryMinor> + Clone

{      
    ///
    /// Construct a `BimajorSparseProductMatrix` struct. 
    /// 
    /// The parameters required to construct an instance of the `BimajorSparseProductMatrix` are identical to those of 
    /// the `BimajorProductMatrix`, as this constructor 
    /// 
    /// 1) creates the lazy product oracle `BimajorProductMatrix`
    /// 2) lazily generates row and column views
    /// 3) respectively compresses them to CSR and CSC representations 
    /// 4) returns the compressed product matrix (CSR) and its transpose (CSC) along with hash map rules for looking up 
    /// integer indexed views given the generic row and column indices of the struct index.
    /// 
    /// This struct is designed to be used as a stand in for the lazy product matrix in applications / use cases where 
    /// repeated lazy operations become too expensive. 
    /// 
    /// It is assumed that the provided vectors of row and column indices are already sorted by their respective order 
    /// operators. 
    /// 
    pub fn new(
        left: LeftMatrix, 
        right: RightMatrix, 
        ring_operator: RingOperator,
        row_indices_of_product: Vec<LeftMatrix::RowIndex>, 
        column_indices_of_product: Vec<RightMatrix::ColIndex>,
        order_operator_key_major_of_product: OrderOperatorKeyMajor, 
        order_operator_view_major_of_product: OrderOperatorViewMajor
    ) -> Self
        where 
            // the where clause is important here, as these are essential for building this struct! 
            LeftMatrix: Clone, 
            RightMatrix: Clone,
            // - the traits below here allow us to take vectors of major or minor entries and cast them to vectors of (index, coef) pairs using an iterator! 
            Vec<(RightMatrix::ColIndex, LeftMatrix::Coefficient)>: FromIterator<RightMatrix::EntryMajor>, 
            Vec<(LeftMatrix::RowIndex, LeftMatrix::Coefficient)>: FromIterator<LeftMatrix::EntryMinor>,
            RightMatrix::ColIndex: PartialOrd, 
            LeftMatrix::RowIndex: PartialOrd, 
    
    { 
        // construct the lazy oracle
        let lazy_product = BimajorProductMatrix::new(
            left,
            right,
            ring_operator.clone(),
            row_indices_of_product.clone(), 
            order_operator_key_major_of_product, 
            order_operator_view_major_of_product.clone() 
        ); 
        // assert that indices are sorted 
        let sorted_row_indices: Vec<LeftMatrix::RowIndex> = lazy_product.sorted_major_keys_of_product.clone(); 
        let is_sorted = column_indices_of_product.is_sorted_by(
            |lhs: &RightMatrix::ColIndex, rhs: &RightMatrix::ColIndex| order_operator_view_major_of_product.judge_partial_cmp(lhs,rhs).unwrap().is_le()
        ); 
        if !is_sorted { 
            panic!("\n\nError: Constructing `BimajorSparseProductMatrix` failed. The list of column indices provided to the constructor is NOT sorted by the accompanying order operator on column indices. \n This message is generated by OAT.\n\n");
        }
        // CSR representation
        let sorted_csr_ascending: Vec<Vec< (RightMatrix::ColIndex, LeftMatrix::Coefficient) >> = row_indices_of_product.clone().into_iter().map(|key|                                             
            lazy_product.view_major_ascend(key).collect() 
        ).collect();
        // CSC representation 
        let sorted_csc_descending: Vec<Vec< (LeftMatrix::RowIndex, LeftMatrix::Coefficient) >> = column_indices_of_product.clone().into_iter().map(|key|                                           
            lazy_product.view_minor_descend(key).collect()
        ).collect();
        // indexing hash maps 
        let row_index_to_integer_bijection: HashMap<LeftMatrix::RowIndex, usize> = sorted_row_indices
            .iter()
            .cloned()
            .enumerate()
            .map(|(x,y)| (y,x) )
            .collect();
        let column_index_to_integer_bijection: HashMap<RightMatrix::ColIndex, usize> = column_indices_of_product
            .iter()
            .cloned()
            .enumerate()
            .map(|(x,y)| (y,x) )
            .collect();
        // construct the matrix
        BimajorSparseProductMatrix { 
            sorted_csr_ascending,
            sorted_csc_descending, 
            sorted_row_indices,
            sorted_column_indices: column_indices_of_product,
            row_index_to_integer_bijection, 
            column_index_to_integer_bijection,
            lazy_product: lazy_product.clone()
        }
    } 
}

/// Implement `IndicesAndCoefficients` for `BimajorSparseProductMatrix`
impl<LeftMatrix, RightMatrix, RingOperator, OrderOperatorKeyMajor, OrderOperatorViewMajor>
    IndicesAndCoefficients for 
        BimajorSparseProductMatrix<LeftMatrix, RightMatrix, RingOperator, OrderOperatorKeyMajor, OrderOperatorViewMajor>
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

/// Implement ViewRowAscend for `BimajorSparseProductMatrix`
impl<LeftMatrix, RightMatrix, RingOperator, OrderOperatorKeyMajor, OrderOperatorViewMajor>
    ViewRowAscend for    
        BimajorSparseProductMatrix<LeftMatrix, RightMatrix, RingOperator, OrderOperatorKeyMajor, OrderOperatorViewMajor>
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
    /// Obtain a major view of the `BimajorSparseProductMatrix`.
    /// 
    fn view_major_ascend(&self, index: Self::RowIndex) -> Self::ViewMajorAscend {
        // NOTE: the wrapped `VecOfVec` matrix assumes that major views are provided in ascending order.
        // thus, all we need to do is map to the integer index and get the row! 
        let integer_index: usize = self.row_index_to_integer_bijection[&index];
        let ref_sorted_csr_ascending = &self.sorted_csr_ascending; 
        let major_view = ref_sorted_csr_ascending[integer_index].clone();
        return major_view;
    }
}

/// Implement ViewColDescend for `BimajorSparseProductMatrix`
impl<LeftMatrix, RightMatrix, RingOperator, OrderOperatorKeyMajor, OrderOperatorViewMajor>
    ViewColDescend for    
        BimajorSparseProductMatrix<LeftMatrix, RightMatrix, RingOperator, OrderOperatorKeyMajor, OrderOperatorViewMajor>
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
    /// Obtain a major view of the `BimajorSparseProductMatrix`.
    /// 
    fn view_minor_descend(&self, index: Self::ColIndex) -> Self::ViewMinorDescend {
        // NOTE: the wrapped `VecOfVec` matrix assumes that minor views are provided in descending order.
        // thus, all we need to do is map to the integer index and get the column! 
        let integer_index: usize = self.column_index_to_integer_bijection[&index];
        let ref_sorted_csc_descending = &self.sorted_csc_descending; 
        let minor_view = ref_sorted_csc_descending[integer_index].clone();
        return minor_view;
    }
}
































//  =========================================================================================================
//  TESTS
//  =========================================================================================================

//  ---------------------------------------------------------------------------
//  DOC-TEST DRAFTS
//  ---------------------------------------------------------------------------

//  We use the following module to draft doc tests, which are easier to debug here than in doc strings.

#[cfg(test)]
mod doc_tests {
    

    use crate::utilities::order::OrderOperatorByLessThan;


    #[test]
    fn doc_test_vector_matrix_product_major_ascend_unsimplified() {
        use crate::algebra::matrices::operations::multiply::vector_matrix_multiply_major_ascend_unsimplified;
        use crate::algebra::matrices::types::vec_of_vec::sorted::VecOfVec;
        use crate::algebra::rings::operator_structs::field_prime_order::PrimeOrderFieldOperator;
        use crate::utilities::order::OrderOperatorByKey;

        // a row-major matrix representation with usize major keys and isize minor keys
        let matrix      =   VecOfVec::new(   vec![   vec![ (0isize, 1), (1isize, 6) ], 
                                                           vec![ (0isize, 1), (1isize, 1) ],     ]     );
        // the vector [1, 1]
        let vector      =   vec![ (0usize, 1), (1usize, 1) ];
        
        // the product
        let product     =   vector_matrix_multiply_major_ascend_unsimplified(
                                vector,
                                & matrix,
                                PrimeOrderFieldOperator::new(7), // the finite field of order 7
                                OrderOperatorByKey::new(), // this compares tuples according to their lexicographic order
                            );
        // the product should return the following sequence of entries: (0,1), (0,1), (1,1), (1,6)   
        itertools::assert_equal( product, vec![(0isize,1usize), (0,1), (1,1), (1,6)] )
    }     

    #[test]
    fn doc_test_vector_matrix_product_major_ascend_simplified() {
        use crate::algebra::matrices::operations::multiply::vector_matrix_multiply_major_ascend_simplified;
        use crate::algebra::matrices::types::vec_of_vec::sorted::VecOfVec;
        use crate::algebra::rings::operator_structs::field_prime_order::PrimeOrderFieldOperator;
        use crate::utilities::order::OrderOperatorByKey;

        // a row-major matrix representation with usize major keys and isize minor keys
        let matrix      =   VecOfVec::new(   vec![   vec![ (0isize, 1), (1isize, 6) ], 
                                                           vec![ (0isize, 1), (1isize, 1) ],     ]     );
        // the vector [1, 1]
        let vector      =   vec![ (0usize, 1), (1usize, 1) ];
        
        // the product
        let product     =   vector_matrix_multiply_major_ascend_simplified(
                                vector,
                                & matrix,
                                PrimeOrderFieldOperator::new(7), // the finite field of order 7
                                OrderOperatorByKey::new(), // this compares tuples according to their lexicographic order
                            );
        // the product should equal vector [2 0]        
        itertools::assert_equal( product, vec![(0isize, 2)] )
    }    

    #[test]
    fn doc_test_vector_matrix_product_minor_descend_unsimplified() {
        use crate::algebra::matrices::operations::multiply::vector_matrix_multiply_minor_descend_unsimplified;
        use crate::algebra::matrices::types::vec_of_vec::sorted::VecOfVec;
        use crate::algebra::rings::operator_structs::field_prime_order::PrimeOrderFieldOperator;
        use crate::utilities::order::OrderOperatorAuto;

        // a row-major matrix representation with usize major keys and isize minor keys
        let matrix      =   VecOfVec::new(   vec![   vec![ (0isize, 1), (1isize, 6) ], 
                                                           vec![ (0isize, 1), (1isize, 1) ],     ]     );
        // the vector [1, 1]
        let vector      =   vec![ (0isize, 1), (1isize, 1) ];
        
        // the product
        let product     =   vector_matrix_multiply_minor_descend_unsimplified(
                                vector,
                                & matrix,
                                PrimeOrderFieldOperator::new(7), // the finite field of order 7
                                OrderOperatorAuto, // this compares tuples according to their lexicographic order
                            );
        // the product should return the following entries in sequence: (1,1), (1,1), (0,1), (0,6)       
        itertools::assert_equal( product, vec![(1,1), (1,1), (0,6), (0,1)] )
    }

    #[test]
    fn doc_test_vector_matrix_product_minor_descend_simplified() {
        use crate::algebra::matrices::operations::multiply::vector_matrix_multiply_minor_descend_simplified;
        use crate::algebra::matrices::types::vec_of_vec::sorted::VecOfVec;
        use crate::algebra::rings::operator_structs::field_prime_order::PrimeOrderFieldOperator;
        use crate::utilities::order::OrderOperatorByKey;

        // a row-major matrix representation with usize major keys and isize minor keys
        let matrix      =   VecOfVec::new(   vec![   vec![ (0isize, 1), (1isize, 6) ], 
                                                           vec![ (0isize, 1), (1isize, 1) ],     ]     );
        // the vector [1, 1]
        let vector      =   vec![ (0isize, 1), (1isize, 1) ];
        
        // the product
        let product     =   vector_matrix_multiply_minor_descend_simplified(
                                vector,
                                & matrix,
                                PrimeOrderFieldOperator::new(7), // the finite field of order 7
                                OrderOperatorByKey::new(), // this compares tuples according to their lexicographic order
                            );
        // the product should equal vector [0 2]        
        itertools::assert_equal( product, vec![(1usize, 2)] )
    }
    
    







    #[test]
    fn doc_test_matrix_product_lazy_major_ascend_unsimplified() {

        // Import the necessary traits and structs
        use crate::algebra::rings::operator_structs::ring_native::DivisionRingNative;
        use crate::algebra::matrices::types::vec_of_vec::sorted::VecOfVec;
        use crate::algebra::matrices::query::{ViewRowAscend};
        use crate::algebra::matrices::operations::multiply::ProductMatrix;
        use crate::algebra::vectors::entries::KeyValGet;
        use std::iter::FromIterator;

        // Define the operator for the coefficient ring
        let ring_operator = DivisionRingNative::<f64>::new();     

        // Define matrix A, a 2x2 matrix with 1's above the diagonal and 0's below
        let matrix_1   =   VecOfVec::new(    vec![ vec![ (0,1.), (1,1.) ], vec![ (1,1.) ] ]    );
        // Define matrix B, a 2x2 matrix with 1's above the diagonal and 0's below
        let matrix_2   =   VecOfVec::new(    vec![ vec![ (0,1.), (1,1.) ], vec![ (1,1.) ] ]    );
        // Define references to the two arrays 
        let matrix_1_ref   =   &matrix_1;
        let matrix_2_ref    =   &matrix_2;        
        // Define the lazy product of A and B
        let product     =   ProductMatrix::new( 
                                    matrix_1_ref,                 // matrix A
                                    matrix_2_ref,                 // matrix B
                                    ring_operator,      // ring operator
                                    OrderOperatorByLessThan::new( |x:&(i32, f64), y:&(i32, f64)| x.key() < y.key() )   // total order on indices
                                );
                            
        // First, check the output iterators themselves (these are wrapped in `Simplify` structs).
        // ---------------------------------------------------------------------------------------
                            
        // Use this object to create a vector-of-vectors that represents the product
        let output =   Vec::from_iter(
                                                (0..2)
                                                    .map(   |x| 
                                                            Vec::from_iter(
                                                                    product
                                                                    .view_major_ascend(x)
                                                                )
                                                        )
                                            );
                                        
        // Check that the answer is correct                                    
        assert_eq!(     
                output,
                vec![ vec![(0,1.), (1,2.)], vec![(1,1.)]  ]
            ); 
        
        // Second, check the underlying, unsimplified iterators (these are are contained inside the `Simplify` structs).
        // -------------------------------------------------------------------------------------------------------------
        
        // Use this object to create a vector-of-vectors that represents the product
        let output =   Vec::from_iter(
                                                (0..2)
                                                    .map(   |x| 
                                                            Vec::from_iter(
                                                                    product
                                                                    .view_major_ascend(x)
                                                                    .unsimplified // this unwraps the inner iterator
                                                                )
                                                        )
                                            );
                                        
        // Check that the answer is correct                                    
        assert_eq!(     
                output,
                vec![ vec![(0,1.), (1,1.), (1,1.)], vec![(1,1.)]  ]
            ); 
    }

}



//  ---------------------------------------------------------------------------
//  TEST
//  ---------------------------------------------------------------------------


#[cfg(test)]
mod tests {
    use std::iter::FromIterator;

    use itertools::Itertools;

    // Note this useful idiom: importing names from outer (for mod tests) scope.
    use super::*;
    use crate::algebra::matrices::types::vec_of_vec::sorted::VecOfVec;
    use crate::algebra::rings::operator_structs::ring_native::{SemiringNative, DivisionRingNative};
    use crate::utilities::order::{OrderOperatorAuto, OrderOperatorByKey, OrderOperatorByLessThan};


    // Test that lazy MATRIX-MATRIX multiplication works properly.
    #[test]
    pub fn matrix_by_matrix_multiply_major_ascend_test_1() {

        // Define the operator for the coefficient ring
        let ring_operator = DivisionRingNative::<f64>::new();     
        
        // Define the factor A, a 2x2 matrix with 1's above the diagonal and 0's below.
        let matrix_1   =   VecOfVec::new(
                                                    // MajorDimension::Row,
                                             vec![ vec![ (0,1.), (1,1.) ], vec![ (1,1.) ] ]
                                                );

        // Define the factor B, a 2x2 matrix with 1's above the diagonal and 0's below.
        let matrix_2   =   VecOfVec::new(
                                                    // MajorDimension::Row,
                                             vec![ vec![ (0,1.), (1,1.) ], vec![ (1,1.) ] ]
                                                );
        let matrix_1_ref    =   &matrix_1;
        let matrix_2_ref    =   &matrix_2;        

        // Define the lazy product of A and B
        let product     =   ProductMatrix::new( 
                                                                        matrix_1_ref, // first borrow is to create something that implements the oracle trait, second borrow is needed to meet requirements of the function syntax
                                                                        matrix_2_ref, 
                                                                        ring_operator, 
                                                                        OrderOperatorByKey::new()
                                                                    );

        // CHECK THE MAJOR ITERATOR ITSELF

        // Use this object to create a vector-of-vectors that represents the product 
        let output =   Vec::from_iter(
                                                (0..2)
                                                    .map(   |x| 
                                                            Vec::from_iter(
                                                                    product
                                                                    .view_major_ascend(x)
                                                                )
                                                        )
                                            );
        
        // Check that the answer is correct                                    
        assert_eq!(     
                output,
                vec![ vec![(0,1.), (1,2.)], vec![(1,1.)]  ]
            );     

        // CHECK THE UNDERLYING **UNSIMPLIFIED** ITERATOR   
        
        // Use this object to create a vector-of-vectors that represents the product 
        let output =   Vec::from_iter(
                                                (0..2)
                                                    .map(   |x| 
                                                            Vec::from_iter(
                                                                    product
                                                                    .view_major_ascend(x)
                                                                    .unsimplified
                                                                )
                                                        )
                                            );
        
        // Check that the answer is correct                                    
        assert_eq!(     
                output,
                vec![ vec![(0,1.), (1,1.), (1,1.)], vec![(1,1.)]  ]
            );           
        
    }


    // Test that lazy MATRIX-MATRIX multiplication works properly - another way.
    #[test]
    pub fn matrix_by_matrix_multiply_major_ascend_test_2() {

        // Define the operator for the coefficient ring
        let ring_operator = SemiringNative::<i32>::new();     
        
        // Define factor A 
        let matrix_1 =   
            VecOfVec::new(
                    vec![ 
                            vec![                  ],
                            vec![ (0,0)            ],                                                            
                            vec![ (0,4),   (1,-1)  ], 
                            vec![ (0,5),   (1,7)   ], 
                            vec![ (0,11),  (1,13)  ],                                                         
                    ]
                );
        // Define factor B 
        let matrix_2 =    
            VecOfVec::new(
                    vec![                                                                                                                           
                            vec![ (0,1),   (1,2),   (2,3) ], 
                            vec![ (0,4),   (1,5),   (2,6) ],
                            vec![                         ],  
                            vec![ (0,0)                   ],                                                              
                        ]
                );
        let matrix_1_ref    =   &matrix_1;
        let matrix_2_ref    =   &matrix_2;  

        // This is the *actual* product A*B, if A and B are row-major.
        let product_true   =   
                    vec![ 
                            vec![                                   ], 
                            vec![                                   ],                                                             
                            vec![          (1,3),     (2,6)         ], 
                            vec![ (0,33),  (1,45),    (2,57)        ],
                            vec![ (0,63),  (1,87),    (2,111)       ],                                                             
                        ];                                              

        // Define the lazy product of A and B
        let product     =   ProductMatrix::new( 
                                                                matrix_1_ref,  // first borrow is to create something that implements the oracle trait, second borrow is needed to meet requirements of the function syntax
                                                                matrix_2_ref, 
                                                                ring_operator, 
                                                                OrderOperatorByLessThan::new(|x:&(i32, i32), y:&(i32, i32)| x.key() < y.key() )  
                                                        );


        // Check that each row of the lazy MATRIX product agrees with each row of the true product.
        for row_index in 0 .. 5 {
            let vec_lazy   =   product
                                                .view_major_ascend( row_index )
                                                .collect_vec();
            let vec_true =  product_true[ row_index ].clone();
            assert_eq!( vec_lazy, vec_true )
        }   
        
    }  
    

   // Test that lazy MATRIX-VECTOR multiplication works properly.
   #[test]
   pub fn matrix_by_vector_multiply_major_ascend_test_2() {

       // Define the operator for the coefficient ring
       let ring_operator = SemiringNative::<i32>::new();     
       
       // Define factor A 
       let matrix_1 =   
           VecOfVec::new(
                   vec![ 
                           vec![                  ],
                           vec![ (0,0)            ],                                                            
                           vec![ (0,4),   (1,-1)  ], 
                           vec![ (0,5),   (1,7)   ], 
                           vec![ (0,11),  (1,13)  ],                                                         
                   ]
               );
       // Define factor B 
       let matrix_2 =    
           VecOfVec::new(
                   vec![                                                                                                                           
                           vec![ (0,1),   (1,2),   (2,3) ], 
                           vec![ (0,4),   (1,5),   (2,6) ],
                           vec![                         ],  
                           vec![ (0,0)                   ],                                                              
                       ]
                );
       // This is the *actual* product A*B, if A and B are row-major.
       let product_true   =   
            vec![ 
                    vec![                                   ], 
                    vec![                                   ],                                                             
                    vec![          (1,3),     (2,6)         ], 
                    vec![ (0,33),  (1,45),    (2,57)        ],
                    vec![ (0,63),  (1,87),    (2,111)       ],                                                             
                ];                                                                             


       // Check that each row of the lazy MATRIX product agrees with each row of the true product.
       for row_index in 0 .. 5 {
           let vec_lazy   =   vector_matrix_multiply_major_ascend_simplified(
                                                    (& matrix_1).view_major_ascend( row_index ).clone(),               
                                                    & matrix_2,  // first borrow is to create something that implements the oracle trait, second borrow is needed to meet requirements of the function syntax
                                                    ring_operator,
                                                    OrderOperatorByLessThan::new( |x:&(i32, i32), y:&(i32, i32)| x.key() < y.key()  )
                                                )
                                               .collect_vec();
           let vec_true =  product_true[ row_index ].clone();
           assert_eq!( vec_lazy, vec_true )
       }   
       
   } 

   // Test that MATRIX-MATRIX multiplication with bimajor views works properly
   // NOTE: `BimajorProductMatrix` is used to build the struct `BimajorSparseProductMatrix`, so this unit test covers both structs!
   #[test]
   pub fn matrix_by_matrix_multiply_bimajor_test() {

        // Define the operator for the coefficient ring
        let ring_operator = SemiringNative::<i32>::new();     
        
        // Define CSR factor A 
        let matrix_1 = VecOfVec::new(
            vec![ 
                        vec![ (0,3), (2,4) ],
                        vec![ (1,-1) ],                                                            
                        vec![ (0,12), (1,2) ], 
                        vec![ (2,7) ],                                                         
                    ]
        );

        // Define CSR factor B 
        let matrix_2 =    VecOfVec::new(
            vec![                                                                                                                           
                        vec![ (0,1), (1,6), (3,7) ], 
                        vec![ (0,-1), (2,3) ],
                        vec![ (1,1), (4,-8) ],                                                          
                    ]
        );

        let matrix_1_ref = &matrix_1;
        let matrix_2_ref  = &matrix_2;  

        // This is the *actual* product A*B, since A and B are row-major.
        let csr_product_true =   vec![                                                             
            vec![ (0,3), (1,22), (3,21), (4,-32) ],
            vec![ (0,1), (2,-3) ],
            vec![ (0,10), (1,72), (2,6), (3,84) ],
            vec![ (1,7), (4,-56) ],
        ];
        let csc_product_true =   vec![                                                             
            vec![ (0,3), (1,1), (2,10) ],
            vec![ (0,22), (2,72), (3,7) ],
            vec![ (1,-3), (2,6) ],
            vec![ (0,21), (2,84) ],
            vec![ (0,-32), (3,-56) ]
        ];                                              

        // Define the lazy product of A and B
        let order_operator_major = OrderOperatorAuto::new(); 
        let order_operator_minor = OrderOperatorAuto::new(); 
        let lazy_product = BimajorProductMatrix::new( 
            matrix_1_ref,  
            matrix_2_ref, 
            ring_operator, 
            vec![0,1,2,3],
            order_operator_major, 
            order_operator_minor
        );

        // Define the sparse product of A and B
        let order_operator_major = OrderOperatorAuto::new(); 
        let order_operator_minor = OrderOperatorAuto::new(); 
        let sparse_product = BimajorSparseProductMatrix::new( 
            matrix_1_ref,  
            matrix_2_ref, 
            ring_operator, 
            vec![0,1,2,3],
            vec![0,1,2,3,4],
            order_operator_major, 
            order_operator_minor
        );

        // Compare the rows of the lazy product with the computed and hardcoded CSR product
        for row_index in 0 .. 4 {
            let vec_lazy = lazy_product
                .view_major_ascend( row_index )
                .collect_vec();
            let vec_sparse = sparse_product.view_major_ascend( row_index ); 
            let vec_true = csr_product_true[ row_index ].clone();
            assert_eq!( vec_lazy, vec_sparse );
            assert_eq!( vec_lazy, vec_true );
        }   

        // Compare the columns of the lazy product with the computed and hardcoded CSC product
        for col_index in 0 .. 5 {
            let vec_lazy= lazy_product
                .view_minor_descend( col_index )
                .collect_vec();
            let vec_sparse= sparse_product.view_minor_descend( col_index );
            let mut vec_true =  csc_product_true[ col_index ].clone();
            vec_true.reverse(); 
            assert_eq!( vec_lazy, vec_sparse );
            assert_eq!( vec_lazy, vec_true );
        }   
   } 
}