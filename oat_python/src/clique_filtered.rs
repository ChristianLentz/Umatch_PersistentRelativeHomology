//!  Filtered clique (Vietoris-Rips) complexes

use num::Signed;
use pyo3::exceptions::{PyGeneratorExit, PyTypeError};
use pyo3::prelude::*;
use pyo3::types::IntoPyDict;
use pyo3::wrap_pyfunction;
// use pyo3_log;
use pyo3::types::{PyDict, PyTuple};
use oat_rust::algebra::chains::factored::FactoredBoundaryMatrix;
use oat_rust::algebra::chains::factored::factor_boundary_matrix;
use oat_rust::algebra::matrices::display::print_indexed_major_views;
use oat_rust::algebra::matrices::types::third_party::IntoCSR;

use oat_rust::algebra::vectors::entries::{KeyValSet, KeyValNew};
use oat_rust::algebra::matrices::operations::multiply::{vector_matrix_multiply_minor_descend_simplified, BimajorProductMatrix};
use oat_rust::algebra::matrices::{operations::umatch::row_major::Umatch, query::{ViewRowAscend, ViewColDescend, IndicesAndCoefficients}};
use oat_rust::algebra::matrices::types::compressed_sparse::{CompressedSparse, lazy_oracle_to_csr_view_minor, lazy_oracle_to_csr_view_major};
use oat_rust::algebra::rings::operator_traits::{Semiring, Ring, DivisionRing};
use oat_rust::algebra::rings::operator_structs::ring_native::{FieldRationalSize, DivisionRingNative};
use oat_rust::algebra::chains::relative::order::{OrderOperatorFullComplexFiltrationSimplices, OrderOperatorSubComplexFiltrationSimplices, RelativeBoundaryMatrixRowIndexOrderOperator}; 
use oat_rust::algebra::chains::relative::oracle::{RelativeBoundaryMatrixOracle, RelativeBoundaryMatrixOracleWrapper};
use oat_rust::algebra::chains::relative::traits::{FactorFromArc, SimplexDiameter, VariableSortOracleKeys}; 
use oat_rust::algebra::chains::to_string::ChainToString;
use oat_rust::algebra::chains::barcode::{Bar, Barcode}; 
use oat_rust::utilities::iterators::general::{RequireStrictAscent, RequireStrictAscentWithPanic};
use oat_rust::utilities::order::{JudgePartialOrder, ReverseOrder};
use oat_rust::utilities::order::{OrderOperatorAuto, OrderOperatorByKey, OrderOperatorByKeyCutsom, IntoReverseOrder};
use oat_rust::algebra::vectors::operations::VectorOperations;

use oat_rust::topology::simplicial::simplices::unfiltered::Simplex;
use oat_rust::topology::simplicial::simplices::filtered::OrderOperatorTwistSimplexFiltered;
use oat_rust::utilities::optimization::minimize_l1::minimize_l1;
use oat_rust::utilities::order::JudgeOrder;
use oat_rust::utilities::iterators::general::PeekUnqualified;
use oat_rust::topology::simplicial::simplices::filtered::SimplexFiltered;
use oat_rust::topology::simplicial::from::graph_weighted::ChainComplexVrFiltered;

use crate::dowker::unique_row_indices;
use crate::export::{Export, ForExport};
use crate::import::import_sparse_matrix;
use crate::simplex_filtered::{SimplexFilteredPy, BarPySimplexFilteredRational, BarcodePySimplexFilteredRational, };

use itertools::Itertools;
use num::rational::Ratio;
use ordered_float::OrderedFloat;
use sprs::{CsMatBase, TriMatBase};

use core::f64;
use std::collections::HashMap;
use std::f32::consts::E;
use std::fmt::Debug;
use std::hash::Hash;
use std::iter::Cloned;
use std::sync::Arc;
use std::time::Instant;
use std::time::Duration;


type FilVal     =   OrderedFloat<f64>;
type RingElt    =   Ratio<isize>;


// type SnzValRational = Ratio< i64 >;
// type RingOperatorRational = DivisionRingNative< SnzValRational >;
// type OrderOperatorSimplexFiltered =   OrderOperatorByKey< 
//                                                 SimplexFiltered< OrderedFloat< f64 > >, 
//                                                 SnzValRational,
//                                                 (
//                                                     SimplexFiltered< OrderedFloat< f64 > >, 
//                                                     SnzValRational,
//                                                 )
//                                             >;

                                         

// #[derive(Clone)]
// #[pyclass(unsendable)]
// pub struct ChainComplexVrFilteredPyRational{
//     umatch:             Option<
//                                 Umatch<  
//                                         &'static ChainComplexVrFiltered< FilVal, SnzValRational, RingOperatorRational>,
//                                         RingOperatorRational, 
//                                         OrderOperatorSimplexFiltered,
//                                         OrderOperatorSimplexFiltered,
//                                     >
//                             >
// }


//  ===========================================================
//  Filtered VR / chain complexes 
//  - cycle representatives: a jordan basis vector corresponding to a birth simplex
//  - Emerson-Escolar indices of persistent cycles 
//  - absolute persistent homology 
//  - barcodes 
//  - optimized cycle representatives 
//  - optimized bounding chains 
//  - python wrappers for boundary matrices and target COMBs
//  - Birth/death indices of Jordan blocks in the persistent Jordan canonical form of a filtered boundary matrix.
//  =========================================================== 


#[pyclass]
pub struct FactoredBoundaryMatrixVr{
    factored:   FactoredBoundaryMatrix<
                        // matrix
                        Arc< ChainComplexVrFiltered<
                                Arc< CsMatBase< FilVal, usize, Vec<usize>, Vec<usize>, Vec<FilVal> > >,
                                FilVal, 
                                RingElt, 
                                DivisionRingNative<RingElt>
                            > >,                         
                        // ChainComplexVrFilteredArc<
                        //         Arc< CsMatBase< FilVal, usize, Vec<usize>, Vec<usize>, Vec<FilVal> > >,
                        //         FilVal, 
                        //         RingElt, 
                        //         DivisionRingNative<RingElt>
                        //     >, 
                        DivisionRingNative< RingElt >, // ring operator
                        OrderOperatorByKeyCutsom< 
                                SimplexFiltered< FilVal >,
                                RingElt,
                                (SimplexFiltered<FilVal>, RingElt),
                                OrderOperatorAuto,
                            >, 
                        SimplexFiltered<FilVal>,
                        ( SimplexFiltered<FilVal>, RingElt ),
                        Vec<SimplexFiltered<OrderedFloat<f64>>>,
                    >
}

#[pymethods]
impl FactoredBoundaryMatrixVr{ 

    /// Construct a Vietoris Rips complex and factor the associated boundary matrix, over the field of rational numbers
    /// 
    /// # Arguments
    /// 
    /// - `dissimilarity_matrix`: a sparse dissimilarity matrix; missing entries will be treated as edges that never enter the filtration
    /// - `homology_dimension_max`: the maximum dimension for which homology is desired
    /// 
    /// Entry `dissimilarity_matrix[i,i]` is regarded as the filtration parameter of vertex `i`.
    /// 
    /// # Returns
    /// 
    /// A `FactoredBoundaryMatrixVr`
    /// 
    /// # Panics
    /// 
    /// Panics if 
    /// - `dissimilarity_matrix` is not symmetric
    /// - there exists an `i` such that entry `[i,i]` is not explicitly stored, but some other entry in row `i` *is* explicitly stored.
    ///   this is because we regard missing entries as having infinite value, rather than zero.
    /// - There exists an `i` such that entry `[i,i]` is strictly greater than some other entry in row `i`
    #[new]
    pub fn new(
            py:                         Python<'_>,
            dissimilarity_matrix:       & PyAny,
            homology_dimension_max:     Option< isize >,            
        ) 
    ->  FactoredBoundaryMatrixVr
    {

        let dissimilarity_matrix = import_sparse_matrix(py, dissimilarity_matrix).ok().unwrap();

        // print_indexed_major_views( && dissimilarity_matrix, 0..dissimilarity_matrix.rows() );

        let npoints = dissimilarity_matrix.rows();

        // let npoints = dissimilarity_matrix.len();

        // let scipy_sparse = py.import("scipy.sparse").ok().unwrap();       
        // let df: Py<PyAny> = pandas.call_method("DataFrame", ( dict, ), None)
        // let scipy = PyModule::new(py, "scipy").ok().unwrap();
        // let sparse = PyModule::new(py, "scipy.sparse").ok().unwrap();
    
        // convert the dissimilarity matrix to type FilVal
        // let dissimilarity_matrix_data
        //     =   dissimilarity_matrix.iter().map(|x| x.iter().cloned().map(|x| OrderedFloat(x)).collect_vec() )
        //         .collect_vec().into_csr( npoints, npoints );
        // let dissimilarity_matrix = Arc::new( dissimilarity_matrix_data );           
        let dissimilarity_matrix = Arc::new( dissimilarity_matrix );                   
        let dissimilarity_max = OrderedFloat(   f64::INFINITY );
        let dissimilarity_min = OrderedFloat( - f64::INFINITY );

        // define the ring operator
        let ring_operator = FieldRationalSize::new();
        // define the chain complex
        let chain_complex_data = ChainComplexVrFiltered::new( dissimilarity_matrix, npoints, dissimilarity_max, dissimilarity_min, ring_operator.clone() );
        // get a reference to the chain complex (needed in order to create certain iterators, due to lifetime bounds)
        // let chain_complex_ref = & chain_complex;   
        let chain_complex = Arc::new( chain_complex_data );
        // define an interator to run over the row indices of the boundary matrix 
        let keymaj_vec = chain_complex.cliques_in_order( homology_dimension_max.unwrap_or(1) );
        // obtain a u-match factorization of the boundary matrix
        let factored = factor_boundary_matrix(
                chain_complex, 
                ring_operator,
                OrderOperatorAuto, 
                keymaj_vec,             
            );   

        // ------------------------
        // println!("row indices used for reduction: {:?}", factored.row_indices().iter().cloned().collect_vec() );

        // let dim_fn = |x: &SimplexFiltered<FilVal> | x.dimension() as isize;
        // let fil_fn = |x: &SimplexFiltered<FilVal> | x.filtration();  
        // println!("{:?}", oat_rust::algebra::chains::barcode::barcode( 
        //     factored.umatch(), 
        //     factored.row_indices().iter().cloned(), 
        //     dim_fn, 
        //     fil_fn, 
        //     true, 
        //     true
        // ));
        // ------------------------        

        return FactoredBoundaryMatrixVr{ factored } // FactoredBoundaryMatrix { umatch, row_indices }
    }

   

    /// Returns the row/column indices of the filtered boundary matrix, in sorted order
    /// 
    /// If the max homology dimension passed by the user when factoring the boundary matrix is `d`, then
    /// the indices include
    /// - every simplex of dimension `<= d`, and 
    /// - every simplex of dimension `d+1` that pairs with a simplex of dimension `d`
    pub fn indices_boundary_matrix( &self ) 
            ->  ForExport<
                    Vec<
                         SimplexFiltered<OrderedFloat<f64>> 
                    >
                >
    {
        let mut row_indices  =   self.factored.row_indices().clone();     
        let matching    =   self.factored.umatch().matching_ref();   

        // we have to reverse the order of row indices (within each dimension) to place things in ascending order
        if row_indices.len() > 0 {
            let mut lhs                     =   0;
            let mut lhs_dim                 =   row_indices[lhs].dimension();
            for  rhs in 0 .. row_indices.len() {
                let rhs_dim                 =   row_indices[rhs].dimension();
                if lhs_dim < rhs_dim { // if we jump a dimension, then reverse the last block of equal-dimension simplices
                    let subset = &mut row_indices[lhs..rhs];
                    subset.reverse(); // reverse the indices of dimension lhs_dim
                    lhs_dim                         =   rhs_dim; // increment the dimension
                    lhs                             =   rhs;
                }
            }
            // the last dimension is an edge case:
            let rhs = row_indices.len();  
            let subset = &mut row_indices[lhs..rhs];
            subset.reverse(); // reverse the indices of dimension lhs_dim
        }    

        // it's possible that some row indices match to some column indices that have dimension > the maximum dimension of
        // any row index.  we don't record max dimension, so we have to calculate it by hand
        if matching.num_pairs() > 0 {

            // the max dimension of any row index; this works because rows are placed in ascending order of dimension
            let max_row_dimension       =   row_indices.last().unwrap().dimension();

            // the max dimension of a matched column is the dimension of the last matched column index that is stored 
            // internally by the matching matrix, because entries in the matching matrix are stored in ascending order 
            // of row index, and rows are stored in ascending order of dimension 
            let matched_columns         =   matching.bimap_min_ref().ord_to_val_vec(); // note this is MINOR keys
            let max_col_dimension       =   matched_columns.last().unwrap().dimension(); 

            if max_row_dimension < max_col_dimension { // in this case we need to include some simplices not indexed by our row indices
                // collect just the top-dimensional column indices
                let mut new_simplices       =   matched_columns.iter().filter(|x| x.dimension() == max_col_dimension).cloned().collect_vec();
                new_simplices.sort(); // sort
                row_indices.extend(new_simplices); // add to the row indices
            }
        }

        return row_indices.export()
    }    


    /// Returns the Emerson-Escolar indices of a persistent cycle.
    pub fn indices_emerson_escolar( 
                &self,
                birth_simplex:                      Vec< u16 >,
            ) ->
                ForExport<
                        Vec<
                            SimplexFiltered<OrderedFloat<f64>> 
                        >
                    >
    {
        // inputs
        let array_matching                  =   self.factored.umatch().matching_ref();        
        let order_operator                  =   self.factored.umatch().order_operator_major_reverse();
        
        // matrix a, vector c, and the dimension function
        let dim_fn = |x: &SimplexFiltered<FilVal> | x.dimension() as isize;
        let obj_fn = |x: &SimplexFiltered<FilVal> | x.filtration().into_inner(); 
        let a = |k: &SimplexFiltered<FilVal>| self.factored.jordan_basis_vector(k.clone()); 
             
        // column b
        let diam = self.factored.umatch().mapping_ref().diameter(&birth_simplex).unwrap();
        let birth_column = SimplexFiltered{ vertices: birth_simplex.clone(), filtration: diam };
        let dimension = birth_column.dimension();
        let b = self.factored.jordan_basis_vector( birth_column.clone() );

        let column_indices = 
                self.factored
                    .indices_escolar_hiraoka( & birth_column, dim_fn, ); // indices of all boundary vectors in the jordan basis

        return column_indices.export()
    }                    


    /// Returns the boundary matrix, formatted as a `scipy.sparse.csr_matrix` with rows/columns
    /// labeled by simplices.
    /// 
    /// The indices for this matrix can be retrieved with `self.indices_boundary_matrix()`
    pub fn boundary_matrix( &self, ) -> 
            ForExport<
                CsMatBase< Ratio<isize>, usize, Vec<usize>, Vec<usize>, Vec<Ratio<isize>> >
            >
        {    

        let row_indices     =   self.indices_boundary_matrix().data;
        let inverse_bijection: HashMap<_,_>   =   row_indices.iter().cloned().enumerate().map(|(x,y)| (y,x) ).collect();
        let mapping             =   self.factored.umatch().mapping_ref();
        let shape                           =   (row_indices.len(), row_indices.len());

        let mut indices_row     =   Vec::new();
        let mut indices_col     =   Vec::new();
        let mut vals     =   Vec::new();

        for index_row in row_indices.iter().cloned() {
            for ( index_col, coefficient ) in mapping.view_major_ascend(index_row.clone()) {
                if inverse_bijection.contains_key( &index_col ) { // we screen out columns that are not in our index set
                    indices_row.push( inverse_bijection[&index_row.clone()].clone() );
                    indices_col.push( inverse_bijection[&index_col        ].clone() );                
                    vals.push( coefficient );
                }

            }
        }

        let mat                 =   TriMatBase::from_triplets(shape, indices_row, indices_col, vals);
        let mat                             =   mat.to_csr();
        return mat.export()

        // let order_operator = self.factored.umatch().order_operator_major();
        // for keyminor in self.factored.umatch().matching_ref().bimap_min_ref().ord_to_val_vec().iter() {
        //     // if keyminor.dimension() > 
        // }
    }


    /// Returns the domain COMB, formatted as a `scipy.sparse.csr_matrix` with rows/columns
    /// labeled by simplices.
    /// 
    /// The indices for this matrix can be retrieved with `self.indices_boundary_matrix()`
    pub fn comb_domain( &self, ) -> 
            ForExport<
                CsMatBase< Ratio<isize>, usize, Vec<usize>, Vec<usize>, Vec<Ratio<isize>> >
            >
        {    

        let row_indices     =   self.indices_boundary_matrix().data;
        let inverse_bijection: HashMap<_,_>   =   row_indices.iter().cloned().enumerate().map(|(x,y)| (y,x) ).collect();
        let comb                            =   self.factored.umatch().comb_domain();
        let shape                           =   (row_indices.len(), row_indices.len());

        let mut indices_row     =   Vec::new();
        let mut indices_col     =   Vec::new();
        let mut vals     =   Vec::new();

        for index_row in row_indices.iter().cloned() {
            for ( index_col, coefficient ) in comb.view_major_ascend(index_row.clone()) {
                if inverse_bijection.contains_key( &index_col ) { // we screen out columns that are not in our index set
                    indices_row.push( inverse_bijection[&index_row.clone()].clone() );
                    indices_col.push( inverse_bijection[&index_col        ].clone() );                
                    vals.push( coefficient );
                }

            }
        }

        let mat                 =   TriMatBase::from_triplets(shape, indices_row, indices_col, vals);
        let mat                             =   mat.to_csr();
        return mat.export()

        // let order_operator = self.factored.umatch().order_operator_major();
        // for keyminor in self.factored.umatch().matching_ref().bimap_min_ref().ord_to_val_vec().iter() {
        //     // if keyminor.dimension() > 
        // }
    }     

    // /// Extract a barcode and a basis of cycle representatives
    // /// 
    // /// Computes the persistent homology of the filtered clique complex (ie VR complex)
    // /// with dissimilarity matrix `dissimilarity_matrix`, over the field of rational numbers.  
    // /// 
    // /// - Edges of weight `>= dissimilarity_max` are excluded.
    // /// - Homology is computed in dimensions 0 through `homology_dimension_max`, inclusive
    // /// 
    // /// Returns: `BarcodePySimplexFilteredRational`
    // pub fn barcode( &self ) -> BarcodePySimplexFilteredRational {
    //     // unpack the factored boundary matrix into a barcode
    //     let dim_fn = |x: &SimplexFiltered<FilVal> | x.dimension() as isize;
    //     let fil_fn = |x: &SimplexFiltered<FilVal> | x.filtration();    
    //     let barcode = oat_rust::algebra::chains::barcode::barcode( 
    //             self.factored.umatch(), 
    //             self.factored.row_indices().iter().cloned(), 
    //             dim_fn, 
    //             fil_fn, 
    //             true, 
    //             true
    //         );
          
    //     return BarcodePySimplexFilteredRational::new( barcode )
    // }


    /// Returns the cycle representative corresponding to a birth simplex.
    pub fn jordan_basis_vector(
                &self,
                column_index:      Vec< u16 >,
            ) 
            -> 
            ForExport<
                    Vec< 
                            ( 
                                    SimplexFiltered< OrderedFloat< f64 > >, 
                                    Ratio< isize >,
                                ) 
                        > 
                >
    {   
        let diam = self.factored.umatch().mapping_ref().diameter(&column_index).unwrap(); // calculate diameter
        let column_index = SimplexFiltered{ vertices: column_index, filtration: diam }; // create a filtered simplex
        self.factored.jordan_basis_vector( column_index ).collect_vec().export()
    }

    /// Extract a barcode and a basis of cycle representatives
    /// 
    /// Computes the persistent homology of the filtered clique complex (ie VR complex)
    /// with dissimilarity matrix `dissimilarity_matrix`, over the field of rational numbers.  
    /// 
    /// - Edges of weight `>= dissimilarity_max` are excluded.
    /// - Homology is computed in dimensions 0 through `homology_dimension_max`, inclusive
    /// 
    /// Returns: a Pandas data frame containing a persistent homology cycle basis, c.f. Hang et al. U-match factorization: sparse homological algebra, lazy cycle representatives, and dualities in persistent (co)homology
    pub fn homology( 
                &self,         
                py:                             Python<'_>,
                return_cycle_representatives:   bool,
                return_bounding_chains:         bool,
        ) 
        -> Py<PyAny> 
        {
        // unpack the factored boundary matrix into a barcode
        let dim_fn = |x: &SimplexFiltered<FilVal> | x.dimension() as isize;
        let fil_fn = |x: &SimplexFiltered<FilVal> | x.filtration();    
        
        let barcode = oat_rust::algebra::chains::barcode::barcode( 
                self.factored.umatch(), 
                self.factored.row_indices().iter().cloned(), 
                dim_fn, 
                fil_fn, 
                return_cycle_representatives, 
                return_bounding_chains
            );
        
        let dict = PyDict::new(py);
        dict.set_item( "id", 
            barcode.bars().iter().map(|x| x.id_number() ).collect_vec() ).ok().unwrap();
        dict.set_item( "dimension", 
            barcode.bars().iter().map(|x| x.birth_column().dimension() ).collect_vec() ).ok().unwrap();            
        dict.set_item( "birth", 
            barcode.bars().iter().map(|x| x.birth_f64() ).collect_vec() ).ok().unwrap();
        dict.set_item( "death", 
            barcode.bars().iter().map(|x| x.death_f64() ).collect_vec() ).ok().unwrap();
        dict.set_item( "birth simplex", 
            barcode.bars().iter().map(|x| x.birth_column().vertices() ).collect_vec() ).ok().unwrap();
        dict.set_item( "death simplex", 
            barcode.bars().iter().map(|x| x.death_column().clone().map(|x| x.vertices().clone() ) ).collect_vec()).ok().unwrap();
        if return_cycle_representatives {
            dict.set_item( "cycle representative", 
                barcode.bars().iter().map(|x| x.cycle_representative().as_ref().unwrap().clone().export() ).collect_vec() ).ok().unwrap();
            dict.set_item( "cycle nnz", 
                barcode.bars().iter().map(|x| x.cycle_representative().as_ref().map(|x| x.len() ) ).collect_vec() ).ok().unwrap();            
        }
        if return_bounding_chains {
            dict.set_item( "bounding chain", 
                barcode.bars().iter().map(|x| x.bounding_chain().as_ref().map(|x| x.clone().export()) ).collect_vec() ).ok().unwrap();                    
            dict.set_item( "bounding nnz", 
                barcode.bars().iter().map(|x| x.bounding_chain().as_ref().map(|x| x.len() ) ).collect_vec() ).ok().unwrap();                                
        }
          
        let pandas = py.import("pandas").ok().unwrap();       
        let df: Py<PyAny> = pandas.call_method("DataFrame", ( dict, ), None)
            .map(Into::into).ok().unwrap();
        df.call_method( py, "set_index", ( "id", ), None)
            .map(Into::into).ok().unwrap()            
    }    


    // /// Optimize a cycle a representative with the Gurobi solver
    // /// 
    // /// As input, the function accepts the `birth_simplex` of a cycle represenative `z` for a bar `b` in persistent homology.
    // /// 
    // /// As output, it returns a cycle `c` which represents the same bar, and is as small as possible
    // /// subject to some standard conditions.  See
    // /// [Minimal Cycle Representatives in Persistent Homology Using Linear Programming: An Empirical Study With User’s Guide](https://www.frontiersin.org/articles/10.3389/frai.2021.681117/full)
    // /// for details.
    // /// 
    // /// Specifically, we employ the "edge loss" method to find a solution `x'` to the problem 
    // /// 
    // /// `minimize Cost(Ax + z)`
    // /// 
    // /// where 
    // ///
    // /// - `x` is unconstrained
    // /// - `z` is a cycle representative for the persistent homology class associated to `birth_simplex`
    // /// - `A` is a matrix whose column space equals the space of all cycles `u` such that (i) `u != z`, (ii) `u` is born no later than `z`, and (iii) `u` dies no later than `z`
    // /// - if `z` is a sum of terms of form `z_s * s`, where `s` is a simplex and `z_s` is a real number,
    // ///   then `Cost(z)` is the sum of the absolute values of the products `z_s * diameter(s)`.
    // /// 
    // /// Returns a data frame containing the optimal cycle, its objective value, the solution `x'` (which is labeled `difference in bounding chains`), etc.
    // /// 
    // /// This method is available when the corresponding bar in persistent homology has a finte right-endpoint.
    // #[cfg(feature = "gurobi")]
    // pub fn optimize_cycle_escolar_hiraoka< 'py >( 
    //             &self,
    //             birth_simplex:    Vec< u16 >,
    //             py: Python< 'py >,
    //         ) -> &'py PyDict { // MinimalCyclePySimplexFilteredRational {
    //     use oat_rust::utilities::optimization::gurobi::optimize_cycle_escolar_hiraoka;
        
    //     let dim_fn = |x: &SimplexFiltered<FilVal> | x.dimension() as isize;
    //     let fil_fn = |x: &SimplexFiltered<FilVal> | x.filtration();    
    //     let obj_fn = |x: &SimplexFiltered<FilVal> | x.filtration().into_inner();  

    //     let diam = self.factored.umatch().mapping_ref().matrix_arc().diameter(&birth_simplex).unwrap();
    //     let birth_column = SimplexFiltered{ vertices: birth_simplex, filtration: diam };
    //     let optimized                   =   optimize_cycle_escolar_hiraoka(
    //                                                 self.factored.umatch(),
    //                                                 self.factored.row_indices().iter().cloned(),
    //                                                 dim_fn,
    //                                                 fil_fn,
    //                                                 obj_fn,
    //                                                 birth_column.clone(),
    //                                                 OrderOperatorTwistSimplexFiltered::new(), // we have to provide the order operator separately
    //                                             ).ok().unwrap();   
    //     let cycle_initial                   =   optimized.cycle_initial().clone();
    //     let cycle_optimal                   =   optimized.cycle_optimal().clone();
    //     let bounding_difference              =   optimized.bounding_difference().clone();
    //     let objective_old               =   optimized.objective_initial().clone();
    //     let objective_min               =   optimized.objective_optimal().clone();

    //     let dict = PyDict::new(py);
    //     dict.set_item( "birth simplex", birth_column.vertices().clone() ).ok().unwrap();        
    //     dict.set_item( "dimension", birth_column.vertices().len() ).ok().unwrap();
    //     dict.set_item( "initial cycle objective value", objective_old ).ok().unwrap();
    //     dict.set_item( "optimal cycle objective value", objective_min ).ok().unwrap();
    //     dict.set_item( "initial cycle nnz", cycle_initial.len() ).ok().unwrap();
    //     dict.set_item( "optimal cycle nnz", cycle_optimal.len() ).ok().unwrap();
    //     dict.set_item( "bounding difference nnz", bounding_difference.len() ).ok().unwrap();        
    //     dict.set_item( "initial cycle", cycle_initial.export() ).ok().unwrap();        
    //     dict.set_item( "optimal cycle", cycle_optimal.export() ).ok().unwrap();
    //     dict.set_item( "difference in bounding chains", bounding_difference.export() ).ok().unwrap();        

    //     return dict
    // }    




    // /// Optimize a bounding chain
    // /// 
    // /// As input, the function accepts the `birth_simplex` of a cycle represenative `z` for a bar `b` in persistent homology.
    // /// 
    // /// As output, it returns a cycle `c` which represents the same bar, and is as small as possible
    // /// subject to some standard conditions.  See
    // /// [Minimal Cycle Representatives in Persistent Homology Using Linear Programming: An Empirical Study With User’s Guide](https://www.frontiersin.org/articles/10.3389/frai.2021.681117/full)
    // /// for details.
    // /// 
    // /// Specifically, we employ the "edge loss" method to find a solution `x'` to the problem 
    // /// 
    // /// `minimize Cost(Ax + z)`
    // /// 
    // /// where 
    // ///
    // /// - `x` is unconstrained
    // /// - `z` is a cycle representative for the persistent homology class associated to `birth_simplex`
    // /// - `A` is a matrix whose column space equals the space of all cycles `u` such that (i) `u != z`, (ii) `u` is born no later than `z`, and (iii) `u` dies no later than `z`
    // /// - if `z` is a sum of terms of form `z_s * s`, where `s` is a simplex and `z_s` is a real number,
    // ///   then `Cost(z)` is the sum of the absolute values of the products `z_s * diameter(s)`.
    // /// 
    // /// Returns a data frame containing the optimal cycle, its objective value, the solution `x'` (which is labeled `difference in bounding chains`), etc.
    // /// 
    // /// This method is available when the corresponding bar in persistent homology has a finte right-endpoint.
    // pub fn optimize_cycle< 'py >( 
    //             &self,
    //             birth_simplex:    Vec< u16 >,
    //             py: Python< 'py >,
    //         ) -> &'py PyDict { // MinimalCyclePySimplexFilteredRational {

    //     // inputs
    //     let array_matching                  =   self.factored.umatch().matching_ref();        
    //     let order_operator                  =   self.factored.umatch().order_operator_major_reverse();
        
    //     // matrix a, vector c, and the dimension function
    //     let dim_fn = |x: &SimplexFiltered<FilVal> | x.dimension() as isize;
    //     let obj_fn = |x: &SimplexFiltered<FilVal> | x.filtration().into_inner(); 
    //     let a = |k: &SimplexFiltered<FilVal>| self.factored.jordan_basis_vector(k.clone()); 
             
    //     // column b
    //     let diam = self.factored.umatch().mapping_ref().matrix_arc().diameter(&birth_simplex).unwrap();
    //     let birth_column = SimplexFiltered{ vertices: birth_simplex, filtration: diam };
    //     let b = self.factored.jordan_basis_vector( birth_column.clone() );  

    //     // column indices of a
    //     let column_indices  =   self.factored.escolar_hiraoka_indices( birth_column.clone(), dim_fn );

    //     // solve
    //     let optimized = oat_rust::utilities::optimization::minimize_l1::minimize_l1(a, b, obj_fn, column_indices).unwrap();

    //     // formatting
    //     let to_ratio = |x: f64| -> Ratio<isize> { Ratio::<isize>::approximate_float(x).unwrap() };
    //     let format_chain = |x: Vec<_>| {
    //         let mut r = x
    //             .into_iter()
    //             .map(|(k,v): (SimplexFiltered<_>,f64) | (k,to_ratio(v)))
    //             .collect_vec();
    //         // r.sort_by( |&(k,v), &(l,u)| order_operator.judge_cmp(&l, &k) );
    //         r.sort_by( |a,b| order_operator.judge_cmp(a, b) );
    //         r
    //     };
        
    //     // optimal solution data
    //     let x =     format_chain( optimized.x().clone() );    
    //     println!("{:?}", &x);    
    //     let cycle_optimal =     format_chain( optimized.y().clone() );
    //     let cycle_initial =     optimized.b().clone();        


    //     // triangles involved
    //     let bounding_difference             =   
    //         x.iter().cloned()
    //         .filter( |x| array_matching.contains_keymaj( &x.0) ) // only take entries for boundaries
    //         .map(|(k,v)| (array_matching.keymaj_to_keymin( &k ).clone().unwrap(),v) )
    //         .multiply_matrix_packet_minor_descend( self.factored.jordan_basis_matrix_packet() )
    //         .collect_vec();

    //     // essential cycles involved
    //     let essential_difference            =   
    //         x.iter().cloned()
    //         .filter( |x| array_matching.lacks_keymin( &x.0 ) ) // only take entries for boundaries
    //         .multiply_matrix_packet_minor_descend( self.factored.jordan_basis_matrix_packet() )
    //         .collect_vec();       

    //     let objective_old               =   optimized.cost_b().clone();
    //     let objective_min               =   optimized.cost_y().clone();

    //     let dict = PyDict::new(py);
    //     dict.set_item( "birth simplex", birth_column.vertices().clone() ).ok().unwrap();        
    //     dict.set_item( "dimension", birth_column.vertices().len() as isize - 1 ).ok().unwrap();
    //     dict.set_item( "initial cycle objective value", objective_old ).ok().unwrap();
    //     dict.set_item( "optimal cycle objective value", objective_min ).ok().unwrap();
    //     dict.set_item( "initial cycle nnz", cycle_initial.len() ).ok().unwrap();
    //     dict.set_item( "optimal cycle nnz", cycle_optimal.len() ).ok().unwrap();
    //     dict.set_item( "initial cycle", cycle_initial.export() ).ok().unwrap();        
    //     dict.set_item( "optimal cycle", cycle_optimal.export() ).ok().unwrap();
    //     dict.set_item( "difference in bounding chains nnz", bounding_difference.len() ).ok().unwrap();         
    //     dict.set_item( "difference in bounding chains", bounding_difference.export() ).ok().unwrap();   
    //     dict.set_item( "difference in essential cycles nnz", essential_difference.len() ).ok().unwrap();                                            
    //     dict.set_item( "difference in essential cycles", essential_difference.export() ).ok().unwrap();

    //     return dict
    // }  


    /// Optimize a cycle representative
    /// 
    /// Specifically, we employ the "edge loss" method to find a solution `x'` to the problem 
    /// 
    /// `minimize Cost(Ax + z)`
    /// 
    /// where 
    ///
    /// - `x` is unconstrained
    /// - `z` is a cycle representative for a (persistent) homology class associated to `birth_simplex`
    /// - `A` is a matrix composed of a subset of columns of the Jordna basis
    /// - `Cost(z)` is the sum of the absolute values of the products `z_s * diameter(s)`.
    /// 
    /// # Arguments
    /// 
    /// - The `birth_simplex` of a cycle represenative `z` for a bar `b` in persistent homology.
    /// - The `problem_type` type for the problem. The optimization procedure works by adding linear
    /// combinations of column vectors from the Jordan basis matrix computed in the factorization.
    /// This argument controls which columns are available for the combination.
    ///   - (default) **"preserve PH basis"** adds cycles which appear strictly before `birth_simplex`
    ///     in the lexicographic ordering on filtered simplex (by filtration, then breaking ties by
    ///     lexicographic order on simplices) and die no later than `birth_simplex`.  **Note** this is
    ///     almost the same as the problem described in [Escolar and Hiraoka, Optimal Cycles for Persistent Homology Via Linear Programming](https://link.springer.com/chapter/10.1007/978-4-431-55420-2_5)
    ///     except that we can include essential cycles, if `birth_simplex` represents an essential class. 
    ///   - **"preserve PH basis (once)"** adds cycles which (i) are distince from the one we want to optimize, and
    ///     (ii) appear (respectively, disappear) no later than the cycle of `birth_simplex`.  This is a looser
    ///     requirement than "preserve PH basis", and may therefore produce a tighter cycle.  Note,
    ///     however, that if we perform this optimization on two or more persistent homology classes in a
    ///     basis of cycle representatives for persistent homology, then the result may not be a
    ///     persistent homology basis.
    ///   - **"preserve homology class"** adds every boundary vector
    ///   - "preserve homology calss (once)" adds every cycle except the one represented by `birth_simplex`
    /// - `check_solution_with_tolerance`: if a value `t` is given for this input, then the solver checks that
    ///   `y_i = (Ax + z)_i` up to a numerical error of `t`, for all `i`
    /// 
    /// # Returns
    /// 
    /// A pandas dataframe containing
    /// 
    /// - `z`, labeled "initial cycle"
    /// - `y`, labeled "optimal cycle"
    /// - `x`, which we separate into two components: 
    ///     - "difference in bounding chains", which is made up of codimension-1 simplices
    ///     - "difference in essential cycles", which is made up of codimension-0 simplices
    /// - The number of nonzero entries in each of these chains
    /// - The objective values of the initial and optimized cycles
    /// 
    /// Features
    /// 
    /// 
    /// 
    /// # Related
    /// 
    /// See
    /// 
    /// - [Escolar and Hiraoka, Optimal Cycles for Persistent Homology Via Linear Programming](https://link.springer.com/chapter/10.1007/978-4-431-55420-2_5)
    /// - [Obayashi, Tightest representative cycle of a generator in persistent homology](https://epubs.siam.org/doi/10.1137/17M1159439)
    /// - [Minimal Cycle Representatives in Persistent Homology Using Linear Programming: An Empirical Study With User’s Guide](https://www.frontiersin.org/articles/10.3389/frai.2021.681117/full)
    /// 
    #[pyo3(signature = (birth_simplex, problem_type,))]
    pub fn optimize_cycle< 'py >( 
                &self,
                birth_simplex:                      Vec< u16 >,
                problem_type:                       Option< &str >,
                py: Python< 'py >,
            ) -> Option< Py<PyAny> > { // MinimalCyclePySimplexFilteredRational {

        // inputs
        let array_matching                  =   self.factored.umatch().matching_ref();        
        let order_operator                  =   self.factored.umatch().order_operator_major_reverse();
        
        // matrix a, vector c, and the dimension function
        let dim_fn = |x: &SimplexFiltered<FilVal> | x.dimension() as isize;
        let obj_fn = |x: &SimplexFiltered<FilVal> | x.filtration().into_inner(); 
        let a = |k: &SimplexFiltered<FilVal>| self.factored.jordan_basis_vector(k.clone()); 
             
        // column b
        let diam = self.factored.umatch().mapping_ref().diameter(&birth_simplex).unwrap();
        let birth_column = SimplexFiltered{ vertices: birth_simplex.clone(), filtration: diam };
        let dimension = birth_column.dimension();
        let b = self.factored.jordan_basis_vector( birth_column.clone() );

        let column_indices = match problem_type.unwrap_or("preserve PH basis") {
            "preserve homology class"    =>  {
                self.factored
                    .indices_boundary() // indices of all boundary vectors in the jordan basis
                    .filter(|x| x.dimension()==dimension ) // of appropriate dimension    
                    .collect_vec()     
            }
            "preserve homology basis (once)"    =>  {
                self.factored
                    .indices_cycle() // indices of all boundary vectors in the jordan basis
                    .filter(|x| (x.dimension()==dimension) && (x.vertices() != &birth_simplex) ) // of appropriate dimension 
                    .collect_vec()           
            }    
            "preserve PH basis (once)"    =>  {
                let mut filtration_order = |x: &SimplexFiltered<OrderedFloat<f64>>,y: &SimplexFiltered<OrderedFloat<f64>>| { x.filtration().cmp(&y.filtration()) };
                self.factored
                    .indices_escolar_hiraoka_relaxed( & birth_column, dim_fn, filtration_order, ) // indices of all boundary vectors in the jordan basis
            }    
            "preserve PH basis"    =>  {
                self.factored
                    .indices_escolar_hiraoka( & birth_column, dim_fn, ) // indices of all boundary vectors in the jordan basis
            }
            _ => {
                println!("");
                println!("");
                println!("Error: problem_type must be one of the following: `preserve homology class`, `preserve homology basis (once)`, `preserve PH basis (once)`, or `preserve PH basis`.");
                println!("This error message is generated by OAT.");
                println!("");
                println!("");
                return None
            }                              
        };

        // solve
        let optimized = oat_rust::utilities::optimization::minimize_l1::minimize_l1(a, b, obj_fn, column_indices).unwrap();

        // formatting
        let to_ratio = |x: f64| -> Ratio<isize> { 
            let frac    =   Ratio::<isize>::approximate_float(x);
            if frac == None { println!("unconvertible float: {:?}", x); }
            frac.unwrap()
        };
        let format_chain = |x: Vec<_>| {
            let mut r = x
                .into_iter()
                .map(|(k,v): (SimplexFiltered<_>,f64) | (k,to_ratio(v)))
                .collect_vec();
            // r.sort_by( |&(k,v), &(l,u)| order_operator.judge_cmp(&l, &k) );
            r.sort_by( |a,b| order_operator.judge_cmp(a, b) );
            r
        };
        
        // optimal solution data
        let x                           =     format_chain( optimized.x().clone() );       
        let cycle_optimal               =     format_chain( optimized.y().clone() );
        let cycle_initial               =     optimized.b().clone();        


        // triangles involved
        let mut bounding_difference     =   
            x.iter()
                .cloned()
                .filter( |x| array_matching.contains_keymaj( &x.0) ) // only take entries for boundaries
                .map(|(k,v)| (array_matching.keymaj_to_keymin( &k ).clone().unwrap(),v) )
                .multiply_matrix_packet_minor_descend( self.factored.jordan_basis_matrix_packet() )
                .collect_vec();
        bounding_difference.reverse(); //    PLACE IN ASECNDING ORDER

        // essential cycles involved
        let mut essential_difference    =   
            x.iter().cloned()
            .filter( |x| array_matching.lacks_keymaj( &x.0 ) ) // only take entries for boundaries
            .multiply_matrix_packet_minor_descend( self.factored.jordan_basis_matrix_packet() )
            .collect_vec();      
        essential_difference.reverse(); //    PLACE IN ASECNDING ORDER 

        let objective_old               =   optimized.cost_b().clone();
        let objective_min               =   optimized.cost_y().clone();

        //  CHECK THE RESULTS
        //  --------------------
        //
        //  * COMPUTE (Ax + z) - y
        //  * ENSURE ALL VECTORS ARE SORTED

        let ring_operator   =   self.factored.umatch().ring_operator();
        let order_operator  =   ReverseOrder::new( OrderOperatorByKey::new() );        

        // We place all iterators in wrappers that check that the results are sorted
        let y   =   RequireStrictAscentWithPanic::new( 
                            cycle_optimal.iter().cloned(),  // sorted in reverse
                            order_operator,                 // judges order in reverse
                        );
        

        let z   =   RequireStrictAscentWithPanic::new( 
                            cycle_initial.iter().cloned(),  // sorted in reverse
                            order_operator,                 // judges order in reverse
                        );                                           
            
        // the portion of Ax that comes from essential cycles;  we have go through this more complicated construction, rather than simply multiplying by the jordan basis matrix, because we've changed basis for the bounding difference chain
        let ax0 =   RequireStrictAscentWithPanic::new( 
                            essential_difference.iter().cloned(),   // sorted in reverse
                            order_operator,                         // judges order in reverse
                        );                  

        // the portion of Ax that comes from non-essential cycles;  we have go through this more complicated construction, rather than simply multiplying by the jordan basis matrix, because we've changed basis for the bounding difference chain
        let ax1
            =   RequireStrictAscentWithPanic::new( 
                    bounding_difference
                        .iter()
                        .cloned()
                        .multiply_matrix_packet_minor_descend(self.factored.umatch().mapping_ref_packet()),  // sorted in reverse
                    order_operator,                 // judges order in reverse
                );  


        let ax_plus_z_minus_y
            =   RequireStrictAscentWithPanic::new( 
                    ax0.peekable()
                        .add(
                                ax1.peekable(),
                                ring_operator,
                                order_operator,
                            )
                        .peekable()
                        .add(
                                z.into_iter().peekable(),
                                ring_operator,
                                order_operator,
                            )
                        .peekable()
                        .subtract(
                                y.into_iter().peekable(),
                                ring_operator,
                                order_operator,
                            ),
                    order_operator,                 
                )
                .collect_vec();      

        let dict = PyDict::new(py);

        // row labels
        dict.set_item(
            "type of chain", 
            vec![
                "initial cycle", 
                "optimal cycle", 
                "difference in bounding chains", 
                "difference in essential chains", 
                "Ax + z - y"
            ]
        ).ok().unwrap();

        // objective costs
        dict.set_item(
            "cost", 
            vec![ 
                Some(objective_old), 
                Some(objective_min), 
                None, 
                None, 
                None, 
            ] 
        ).ok().unwrap(); 

        // number of nonzero entries per vector       
        dict.set_item(
            "nnz", 
            vec![ 
                cycle_initial.len(), 
                cycle_optimal.len(), 
                bounding_difference.len(), 
                essential_difference.len(),
                ax_plus_z_minus_y.len(),
            ] 
        ).ok().unwrap();

        // vectors
        dict.set_item(
            "chain", 
            vec![ 
                cycle_initial.clone().export(), 
                cycle_optimal.clone().export(), 
                bounding_difference.clone().export(), 
                essential_difference.clone().export(),
                ax_plus_z_minus_y.clone().export(),
                ] 
        ).ok().unwrap();   

        let pandas = py.import("pandas").ok().unwrap();       
        let dict = pandas.call_method("DataFrame", ( dict, ), None)
            .map(Into::< Py<PyAny> >::into).ok().unwrap();
        let kwarg = vec![("inplace", true)].into_py_dict(py);        
        dict.call_method( py, "set_index", ( "type of chain", ), Some(kwarg)).ok().unwrap();        

        return Some( dict )
    }        




    #[pyo3(signature = (birth_simplex,))]
    pub fn optimize_bounding_chain< 'py >( 
                &self,
                birth_simplex:                      Vec< u16 >,
                py: Python< 'py >,
            ) -> Option< Py<PyAny> > { // MinimalCyclePySimplexFilteredRational {

        // inputs
        let array_mapping                   =   self.factored.umatch().mapping_ref_packet();                        
        let array_matching                  =   self.factored.umatch().matching_ref();    
        let order_operator                  =   self.factored.umatch().order_operator_major_reverse();
        
        // matrix a, vector c, and the dimension function
        let dim_fn = |x: &SimplexFiltered<FilVal> | x.dimension() as isize;
        let obj_fn = |x: &SimplexFiltered<FilVal> | x.filtration().into_inner(); 
        let a = |k: &SimplexFiltered<FilVal>| self.factored.jordan_basis_vector(k.clone()); 
             
        // column b
        let diam = self.factored.umatch().mapping_ref().diameter(&birth_simplex).unwrap();
        let birth_column = SimplexFiltered{ vertices: birth_simplex.clone(), filtration: diam };

        if array_matching.lacks_keymaj( &birth_column ) {
            println!("\n\nError: the birth simplex provided has no corresponding death simplex.\nThis message is generated by OAT.\n\n");
            return None
        }

        let death_column        =   array_matching.keymaj_to_keymin( &birth_column ).unwrap();
        let death_dimension     =   death_column.dimension();
        let death_filtration    =   death_column.filtration();
        let b                   =   self.factored.jordan_basis_vector( death_column.clone() );

        // incides of a
        let column_indices
            =   self.factored.umatch().mapping_ref()
                    .cliques_in_lexicographic_order_fixed_dimension( death_dimension as isize )
                    .filter(
                        |x|
                        ( x.filtration() <=   death_filtration )
                        &&
                        ( ! array_matching.contains_keymin(&x) ) // exclude positive simplices; in particular, this excluds the death simplex
                    );

        // solve
        let optimized = oat_rust::utilities::optimization::minimize_l1::minimize_l1(a, b, obj_fn, column_indices).unwrap();

        // formatting
        let to_ratio = |x: f64| -> Ratio<isize> { Ratio::<isize>::approximate_float(x).unwrap() };
        let format_chain = |x: Vec<_>| {
            let mut r = x
                .into_iter()
                .map(|(k,v): (SimplexFiltered<_>,f64) | (k,to_ratio(v)))
                .collect_vec();
            // r.sort_by( |&(k,v), &(l,u)| order_operator.judge_cmp(&l, &k) );
            r.sort_by( |a,b| order_operator.judge_cmp(a, b) );
            r
        };

        // optimal solution data     
        let chain_optimal               =     format_chain( optimized.y().clone() );
        let mut chain_initial           =     optimized.b().clone();        
        chain_initial.sort();

        let objective_old               =   optimized.cost_b().clone();
        let objective_min               =   optimized.cost_y().clone();


        //  CHECK THE RESULTS
        //  --------------------

        let boundary_initial            =   chain_initial.iter().cloned().multiply_matrix_packet_minor_descend( array_mapping.clone() ).collect_vec();
        let boundary_optimal            =   chain_optimal.iter().cloned().multiply_matrix_packet_minor_descend( array_mapping.clone() ).collect_vec();
        let diff                        =   boundary_initial.iter().cloned().peekable().subtract(
                                                boundary_optimal.iter().cloned().peekable(),
                                                self.factored.umatch().ring_operator(),
                                                self.factored.umatch().order_operator_major_reverse(),
                                            )
                                            .map( |x| x.1.abs() )
                                            .max();
        println!("max difference in boundaries: {:?}", diff);
        // assert_eq!( boundary_initial, boundary_optimal ); // ensures that the initial and optimal chains have equal boundary


        let dict = PyDict::new(py);

        // row labels
        dict.set_item(
            "type of chain", 
            vec![
                "initial bounding chain", 
                "optimal bounding chain", 
            ]
        ).ok().unwrap();

        // objective costs
        dict.set_item(
            "cost", 
            vec![ 
                Some(objective_old), 
                Some(objective_min), 
            ] 
        ).ok().unwrap(); 

        // number of nonzero entries per vector       
        dict.set_item(
            "nnz", 
            vec![ 
                chain_initial.len(), 
                chain_optimal.len(), 
            ] 
        ).ok().unwrap();

        // vectors
        dict.set_item(
            "chain", 
            vec![ 
                chain_initial.clone().export(), 
                chain_optimal.clone().export(), 
                ] 
        ).ok().unwrap();   

        let pandas = py.import("pandas").ok().unwrap();       
        let dict = pandas.call_method("DataFrame", ( dict, ), None)
            .map(Into::< Py<PyAny> >::into).ok().unwrap();
        let kwarg = vec![("inplace", true)].into_py_dict(py);        
        dict.call_method( py, "set_index", ( "type of chain", ), Some(kwarg)).ok().unwrap();            

        return Some( dict )
    }        



    /// Minimize a bounding chain, subject to the constraint that the new bounding chain
    /// has equal boundary, and is obtained by adding simplices born strictly before the
    /// last simplex in the input bounding chain in the refined filtration order.
    /// 
    /// The constraint is formalized as "minimize l1 norm of `b + x`,subject to the condition that `Dx = 0`".
    /// 
    /// We suspect this is less efficient that other formulations.
    #[pyo3(signature = (birth_simplex, problem_type,))]
    pub fn optimize_bounding_chain_kernel< 'py >( 
                &self,
                birth_simplex:                      Vec< u16 >,
                problem_type:                       Option< &str >,
                py: Python< 'py >,
            ) -> Option< Py<PyAny> > { // MinimalCyclePySimplexFilteredRational {

        // inputs
        let array_mapping                   =   self.factored.umatch().mapping_ref_packet();        
        let array_matching                  =   self.factored.umatch().matching_ref();        
        let order_operator                  =   self.factored.umatch().order_operator_major_reverse();
        
        // matrix a, vector c, and the dimension function
        let dim_fn = |x: &SimplexFiltered<FilVal> | x.dimension() as isize;
        let obj_fn = |x: &SimplexFiltered<FilVal> | x.filtration().into_inner(); 
        let a = |k: &SimplexFiltered<FilVal>| array_mapping.matrix.view_minor_descend( k.clone() ); // columns of A are columns of the boundary matrix
             
        // column b
        let diam = self.factored.umatch().mapping_ref().diameter(&birth_simplex).unwrap();
        let birth_column = SimplexFiltered{ vertices: birth_simplex.clone(), filtration: diam };

        if array_matching.lacks_keymaj( &birth_column ) {
            println!("\n\nError: the birth simplex provided has no corresponding death simplex.\nThis message is generated by OAT.\n\n");
            return None
        }

        let death_column        =   array_matching.keymaj_to_keymin( &birth_column ).unwrap();
        let death_dimension     =   death_column.dimension();
        let death_filtration    =   death_column.filtration();
        let b                   =   self.factored.jordan_basis_vector( death_column.clone() ).collect_vec();

        let column_indices = match problem_type.unwrap_or("preserve PH basis") {
            "preserve PH basis"    =>  {

                Some( 
                        self.factored.umatch().mapping_ref()
                            .cliques_in_lexicographic_order_fixed_dimension( death_dimension as isize )
                            .filter(
                                |x|
                                ( x.filtration() <=   death_filtration )
                                &&
                                ( x != &death_column )
                            )  
                    )
            }
            _ => {
                println!("");
                println!("");
                println!("Error: problem_type must be one of the following: `preserve homology class`, `preserve homology basis (once)`, `preserve PH basis (once)`, or `preserve PH basis`.");
                println!("This error message is generated by OAT.");
                println!("");
                println!("");
                None
            }                              
        }?;

        // solve
        let optimized = oat_rust::utilities::optimization::minimize_l1::minimize_l1_kernel(a, b, obj_fn, column_indices).unwrap();

        // formatting
        let to_ratio = |x: f64| -> Ratio<isize> { Ratio::<isize>::approximate_float(x).unwrap() };
        let format_chain = |x: Vec<_>| {
            let mut r = x
                .into_iter()
                .map(|(k,v): (SimplexFiltered<_>,f64) | (k,to_ratio(v)))
                .collect_vec();
            // r.sort_by( |&(k,v), &(l,u)| order_operator.judge_cmp(&l, &k) );
            r.sort_by( |a,b| order_operator.judge_cmp(a, b) );
            r
        };
        
        // optimal solution data
        let x                           =     format_chain( optimized.x().clone() );       
        let chain_optimal               =     format_chain( optimized.y().clone() );
        let mut chain_initial           =     optimized.b().clone();        
        chain_initial.sort();

        let objective_old               =   optimized.cost_b().clone();
        let objective_min               =   optimized.cost_y().clone();


        //  CHECK THE RESULTS
        //  --------------------

        // let boundary_initial            =   chain_initial.iter().cloned().multiply_matrix_packet_minor_descend( array_mapping.clone() ).collect_vec();
        // let boundary_optimal            =   chain_optimal.iter().cloned().multiply_matrix_packet_minor_descend( array_mapping.clone() ).collect_vec();
        // assert_eq!( boundary_initial, boundary_optimal ); // ensures that the initial and optimal chains have equal boundary


        let dict = PyDict::new(py);

        // row labels
        dict.set_item(
            "type of chain", 
            vec![
                "initial bounding chain", 
                "optimal bounding chain", 
                "difference in bounding chains", 
            ]
        ).ok().unwrap();

        // objective costs
        dict.set_item(
            "cost", 
            vec![ 
                Some(objective_old), 
                Some(objective_min), 
                None, 
            ] 
        ).ok().unwrap(); 

        // number of nonzero entries per vector       
        dict.set_item(
            "nnz", 
            vec![ 
                chain_initial.len(), 
                chain_optimal.len(), 
                x.len(), 
            ] 
        ).ok().unwrap();

        // vectors
        dict.set_item(
            "chain", 
            vec![ 
                chain_initial.clone().export(), 
                chain_optimal.clone().export(), 
                x.clone().export(), 
                ] 
        ).ok().unwrap();   

        let pandas = py.import("pandas").ok().unwrap();       
        let dict = pandas.call_method("DataFrame", ( dict, ), None)
            .map(Into::< Py<PyAny> >::into).ok().unwrap();
        let kwarg = vec![("inplace", true)].into_py_dict(py);        
        dict.call_method( py, "set_index", ( "type of chain", ), Some(kwarg)).ok().unwrap();        

        return Some( dict )
    }        









    /// Returns the birth/death row/column indices of the Jordan blocks in the persistent Jordan canonical form of the filtered boundary matrix.
    pub fn jordan_block_indices< 'py >( &self,  py: Python< 'py >, ) -> Py<PyAny> {
        // unpack the factored boundary matrix into a barcode
        let dim_fn = |x: &SimplexFiltered<FilVal> | x.dimension() as isize;
        let fil_fn = |x: &SimplexFiltered<FilVal> | x.filtration(); 
        let matching = self.factored.umatch().matching_ref();
        let births = self.factored.indices_cycle();

        let mut death_simplex_opt:      Option< SimplexFiltered<OrderedFloat<f64>> >;
        let mut dimension               = Vec::new(); 
        let mut birth_vertices          = Vec::new(); 
        let mut birth_filtration        = Vec::new(); 
        let mut death_vertices_opt      = Vec::new(); 
        let mut death_filtration_opt    = Vec::new();
        let mut lifetime                = Vec::new();
        
        for keymaj in self.factored.indices_cycle() {
            dimension.push( keymaj.dimension() );
            birth_vertices.push( keymaj.vertices().clone() );
            birth_filtration.push( keymaj.filtration().into_inner() );
            death_simplex_opt = matching.keymaj_to_keymin( & keymaj).map(|x| x.clone()); //.Option::<SimplexFiltered<OrderedFloat<f64>>>::cloned();
            death_vertices_opt.push( death_simplex_opt.clone().map(|x| x.vertices().clone() ) );
            death_filtration_opt.push( death_simplex_opt.map(|x| x.filtration().into_inner() ) );
            lifetime.push(
                match death_filtration_opt[ death_filtration_opt.len() - 1] {
                    Some( f ) => { f - birth_filtration[birth_filtration.len()-1] }
                    None => { std::f64::INFINITY }
                }
            );
        }

        // let to_death_vertices_opt = |keymaj: &SimplexFiltered<_>|  { 
        //     matching.keymaj_to_keymin(keymaj).cloned().map(|x| x.vertices() )
        // };
        // let to_death_filtration_opt = |keymaj: &SimplexFiltered<_>|  { 
        //     matching.keymaj_to_keymin(keymaj).cloned().map(|x| x.filtration().into_inner() )
        // };

        // let pairs = self.factored.umatch().matching_ref().support();
        
        let dict = PyDict::new(py);
        dict.set_item( "dimension", dimension ).ok().unwrap();
        dict.set_item( "lifetime", lifetime ).ok().unwrap();        
        dict.set_item( "birth simplex", birth_vertices ).ok().unwrap();        
        dict.set_item( "birth filtration", birth_filtration ).ok().unwrap();         
        dict.set_item( "death simplex", death_vertices_opt ).ok().unwrap();
        dict.set_item( "death filtration", death_filtration_opt ).ok().unwrap();         
        // dict.set_item( "birth simplex", 
        //     births.clone().map(|x| x.0.vertices() ).collect_vec() ).ok().unwrap();
        // dict.set_item( "birth filtration", 
        //     births.clone().map(|x| x.0.filtration().into_inner() ).collect_vec() ).ok().unwrap(); 
        // dict.set_item( "death simplex", 
        //     births.clone().map( to_death_vertices_opt ).collect_vec() ).ok().unwrap();
        // dict.set_item( "death filtration", 
        //     births.clone().map( to_death_filtration_opt ).collect_vec() ).ok().unwrap(); 

        let pandas = py.import("pandas").ok().unwrap();       
        pandas.call_method("DataFrame", ( dict, ), None)
            .map(Into::into).ok().unwrap()
        // df.call_method( py, "set_index", ( "id", ), None)
        //     .map(Into::into).ok().unwrap()           
    }    

    // pub fn test_subcloud_indices_fn(
    //     &self,       
    //     py: Python<'_>,
    //     subcloud_indices: Vec<usize>
    // ) -> Py<PyAny>
    
    // { 
    //     // lists we need to build the hash map 
    //     let zero_simplices = self.factored.umatch().mapping_ref().cliques_in_order(0);
    //     // construct a hash map H from zero simplices to {0,1}, with 1 indicating that the zero simplex is in fact in the subcomplex. 
    //     let mut map = HashMap::with_capacity(zero_simplices.clone().len());
    //     let mut iter_subcloud_indices = subcloud_indices.iter();
    //     let mut next_idx = iter_subcloud_indices.next();
    //     // for each zero simplex in the complex (in ascending lexicographic order)
    //     for (i, val) in zero_simplices.iter().rev().enumerate() {
    //         // if it maps to a subcloud index
    //         if Some(&i) == next_idx {
    //             map.insert(val.vertices[0].clone(), 1);
    //             next_idx = iter_subcloud_indices.next(); // --> increment the iterator over subcloud indices
    //         // if it does not map to a subcloud index
    //         } else {
    //             map.insert(val.vertices[0].clone(), 0);
    //         }
    //     }

    //     let dict = PyDict::new(py);
    //     dict.set_item( "id", 
    //         zero_simplices.iter().map(|x| map[&x.vertices[0]] ).collect_vec() ).ok().unwrap();
    //     dict.set_item( "simplex", 
    //         zero_simplices.iter().map(|x| x.vertices_to_string() ).collect_vec() ).ok().unwrap();            
    //     dict.set_item( "inSubcomplex", 
    //         zero_simplices.iter().map(|x| map[&x.vertices[0]] ).collect_vec() ).ok().unwrap();
    //     let pandas = py.import("pandas").ok().unwrap();       
    //     let df: Py<PyAny> = pandas.call_method("DataFrame", ( dict, ), None).map(Into::into).ok().unwrap();
    //     df.call_method( py, "set_index", ( "id", ), None).map(Into::into).ok().unwrap() 
    // }

    ///
    /// Computes persistent relative homology of the point cloud under the assumption that the subcomplex filtration 
    /// is identical to the full complex filtration up to some nonnegative real number d. For example, if X is the VR 
    /// complex and f is a sublevel set filtration of the complex, then the sublevel set filtration for the subcomplex 
    /// is of the form g(x) = f(x) + d. 
    /// 
    /// This special case allows PRH to be computed in a single decomposition step, and it's implemented here as a simple 
    /// and efficient alternative to the methods (currently under construction) found in `oat_rust/src/algebra/chains/realtive`.
    /// 
    /// # Parameters
    /// 
    /// - `delta` -> the shift or lag in the subcomplex filtration.
    /// - `return_cycle_representatives` -> a boolean determining whether cycle representatives are returned. 
    /// - `return_bounding_chains` -> a boolean determining whether bounding chains are returned. 
    /// 
    pub fn persistent_relative_homology_lag_filtration(
        &self,       
        py: Python<'_>,
        delta: f64, 
        return_cycle_representatives: bool, 
        return_bounding_chains: bool
    ) -> Py<PyAny>
    
    {
        let dim_fn = |x: &SimplexFiltered<FilVal> | x.dimension() as isize;
        let fil_fn = |x: &SimplexFiltered<FilVal> | x.filtration();   
        
        // build the barcode
        let barcode = oat_rust::algebra::chains::barcode::barcode_relative_homology_lag_filtration( 
            self.factored.umatch(), 
            self.factored.row_indices().iter().cloned(), 
            dim_fn, 
            fil_fn, 
            return_cycle_representatives, 
            return_bounding_chains,
            delta, 
        );
        
        // unpack the barcode and send to python
        let dict = PyDict::new(py);
        dict.set_item( "id", 
            barcode.bars().iter().map(|x| x.id_number() ).collect_vec() ).ok().unwrap();
        dict.set_item( "dimension", 
            barcode.bars().iter().map(|x| x.birth_column().dimension() ).collect_vec() ).ok().unwrap();            
        dict.set_item( "birth", 
            barcode.bars().iter().map(|x| x.birth_f64() ).collect_vec() ).ok().unwrap();
        dict.set_item( "death", 
            barcode.bars().iter().map(|x| x.death_f64() ).collect_vec() ).ok().unwrap();
        dict.set_item( "birth simplex", 
            barcode.bars().iter().map(|x| x.birth_column().vertices() ).collect_vec() ).ok().unwrap();
        dict.set_item( "death simplex", 
            barcode.bars().iter().map(|x| x.death_column().clone().map(|x| x.vertices().clone() ) ).collect_vec()).ok().unwrap();
        if return_cycle_representatives {
            dict.set_item( "cycle representative", 
                barcode.bars().iter().map(|x| x.cycle_representative().as_ref().unwrap().clone().export() ).collect_vec() ).ok().unwrap();
            dict.set_item( "cycle nnz", 
                barcode.bars().iter().map(|x| x.cycle_representative().as_ref().map(|x| x.len() ) ).collect_vec() ).ok().unwrap();            
        }
        if return_bounding_chains {
            dict.set_item( "bounding chain", 
                barcode.bars().iter().map(|x| x.bounding_chain().as_ref().map(|x| x.clone().export()) ).collect_vec() ).ok().unwrap();                    
            dict.set_item( "bounding nnz", 
                barcode.bars().iter().map(|x| x.bounding_chain().as_ref().map(|x| x.len() ) ).collect_vec() ).ok().unwrap();                                
        }
        let pandas = py.import("pandas").ok().unwrap();       
        let df: Py<PyAny> = pandas.call_method("DataFrame", ( dict, ), None).map(Into::into).ok().unwrap();
        df.call_method( py, "set_index", ( "id", ), None).map(Into::into).ok().unwrap() 
    }

}

// #[pyfunction]
// pub fn get_factored_vr_complex( 
//             dissimilarity_matrix: Vec<Vec<f64>>, 
//             homology_dimension_max: isize,
//             dissimilarity_max: Option<f64>,
//         ) 
//     ->  FactoredBoundaryMatrixVr
//     // FactoredBoundaryMatrix<
//     //             OracleDeref< 
//     //                     ChainComplexVrFiltered<
//     //                             OracleDeref< CsMatBase< FilVal, usize, Vec<usize>, Vec<usize>, Vec<FilVal> > >,
//     //                             FilVal, 
//     //                             RingElt, 
//     //                             DivisionRingNative<RingElt>
//     //                         >                
//     //                 >,
//     //             DivisionRingNative< RingElt >,
//     //             OrderOperatorByKey< 
//     //                     SimplexFiltered< FilVal >,
//     //                     RingElt,
//     //                     ( SimplexFiltered< FilVal >, RingElt ) 
//     //                 >,
//     //             SimplexFiltered< FilVal >,
//     //             ( SimplexFiltered< FilVal >, RingElt ) ,
//     //             Cloned<Iter<SimplexFiltered<FilVal>>>
//     //         >
//     {

//     println!("TODO: retool barcode to return a dataframe");      

//     let npoints = dissimilarity_matrix.len();  
 
//     // convert the dissimilarity matrix to type FilVal
//     let dissimilarity_matrix_data
//         =   dissimilarity_matrix.iter().map(|x| x.iter().cloned().map(|x| OrderedFloat(x)).collect_vec() )
//             .collect_vec().into_csr( npoints, npoints );
//     let dissimilarity_matrix = Arc::new( dissimilarity_matrix_data );           
//     let dissimilarity_max = dissimilarity_max.map(|x| OrderedFloat(x));
//     let dissimilarity_min = 
//             { 
//                 if npoints==0 {
//                     OrderedFloat(0.0)
//                 } else { 
//                     dissimilarity_matrix.data().iter().min().unwrap().clone()
//                 } 
//             };               
//     // define the ring operator
//     let ring_operator = FieldRationalSize::new();
//     // define the chain complex
//     let chain_complex_data = ChainComplexVrFiltered::new( dissimilarity_matrix, npoints, dissimilarity_max, dissimilarity_min, ring_operator.clone() );
//     // get a reference to the chain complex (needed in order to create certain iterators, due to lifetime bounds)
//     // let chain_complex_ref = & chain_complex;   
//     let chain_complex = ChainComplexVrFilteredArc::new( Arc::new( chain_complex_data ) );
//     // define an interator to run over the row indices of the boundary matrix 
//     let keymaj_vec = chain_complex.vr().cliques_in_order( homology_dimension_max );
//     let iter_keymaj = keymaj_vec.iter().cloned();    
//     // obtain a u-match factorization of the boundary matrix
//     let factored = factor_boundary_matrix(
//             chain_complex, 
//             ring_operator.clone(),             
//             OrderOperatorAutoLt::new(), 
//             keymaj_vec,             
//         );      
//     return FactoredBoundaryMatrixVr{ factored } // FactoredBoundaryMatrix { umatch, row_indices }

// }




















//  =========================================
//  COMPUTE PERSISTENT HOMOLOGY
//  =========================================


//  DEPRECATED
// /// Compute basis of cycle representatives for persistent homology of a VR filtration, over the rationals
// /// 
// /// Computes the persistent homology of the filtered clique complex (ie VR complex)
// /// with dissimilarity matrix `dissimilarity_matrix`, over the field of rational numbers.  
// /// 
// /// - Edges of weight `>= dissimilarity_max` are excluded.
// /// - Homology is computed in dimensions 0 through `homology_dimension_max`, inclusive
// /// 
// /// Returns: `BarcodePySimplexFilteredRational`
// #[pyfunction]
// pub fn persistent_homology_vr( 
//             dissimilarity_matrix: Vec<Vec<f64>>, 
//             homology_dimension_max: isize,
//             dissimilarity_max: Option<f64>,  
//         ) 
//     -> BarcodePySimplexFilteredRational
//     {

//     println!("TODO: shift to umatch and retool barcode to return a dataframe");
 
//     // // convert the dissimilarity matrix to type FilVal
//     // let dissimilarity_matrix = dissimilarity_matrix.iter().map(|x| x.iter().cloned().map(|x| OrderedFloat(x)).collect() ).collect();
//     // let dissimilarity_max = dissimilarity_max.map(|x| OrderedFloat(x));
//     // // define the ring operator
//     // let ring_operator = FieldRationalSize::new();
//     // // define the chain complex
//     // let chain_complex = ChainComplexVrFiltered::new( dissimilarity_matrix, dissimilarity_max, ring_operator.clone() );
//     // // get a reference to the chain complex (needed in order to create certain iterators, due to lifetime bounds)
//     // let chain_complex_ref = & chain_complex;   
//     // // define an interator to run over the row indices of the boundary matrix 
//     // let keymaj_vec = chain_complex.cliques_in_order( homology_dimension_max );
//     // let iter_keymaj = keymaj_vec.iter().cloned();    
//     // // obtain a u-match factorization of the boundary matrix
//     // let umatch = new_umatchrowmajor_with_clearing(
//     //         chain_complex_ref, 
//     //         iter_keymaj.clone(), 
//     //         ring_operator.clone(), 
//     //         OrderOperatorAutoLt::new(), 
//     //         OrderOperatorAutoLt::new(), 
//     //     ); 

//     let dissimilarity_matrix_data = dissimilarity_matrix.iter().map(|x| x.iter().cloned().map(|x| OrderedFloat(x)).collect_vec() ).collect_vec();
//     let dissimilarity_matrix_size = dissimilarity_matrix.len();
//     let dissimilarity_max = dissimilarity_max.map(|x| OrderedFloat(x));
//     let dissimilarity_min = 
//             { 
//                 if dissimilarity_matrix_size==0 {
//                     OrderedFloat(0.0)
//                 } else { 
//                     dissimilarity_matrix_data.iter().map(|x| x.iter()).flatten().min().unwrap().clone()
//                 } 
//             };
//     let dissimilarity_matrix = Arc::new( dissimilarity_matrix_data.into_csr( dissimilarity_matrix_size, dissimilarity_matrix_size ) );
//     // define the ring operator
//     let ring_operator = FieldRationalSize::new();
//     // define the chain complex
//     let chain_complex = ChainComplexVrFiltered::new( dissimilarity_matrix, dissimilarity_matrix_size, dissimilarity_max, dissimilarity_min, ring_operator.clone() );
//     let factored = chain_complex.factor_from_arc( homology_dimension_max );    

//     // unpack the factored boundary matrix into a barcode
//     let dim_fn = |x: &SimplexFiltered<FilVal> | x.dimension() as isize;
//     let fil_fn = |x: &SimplexFiltered<FilVal> | x.filtration();    
//     let barcode = oat_rust::algebra::chains::barcode::barcode( factored.umatch(), factored.factored.cliques_in_order(homology_dimension_max), dim_fn, fil_fn, true , true);
      
//     return BarcodePySimplexFilteredRational::new( barcode )
// }




//  DEPRECATED IN FAVOR OF METHODS ON A FACTORED COMPLEX
// /// persistent_homology_vr_optimized(dissimilarity_matrix: Vec<Vec<f64>>, homology_dimension_max: isize, dissimilarity_max: Option<f64>, /)
// /// --
// ///
// /// Compute basis of *optimized* cycle representatives for persistent homology of a VR filtration, over the rationals
// /// 
// /// Computes the persistent homology of the filtered clique complex (ie VR complex)
// /// with dissimilarity matrix `dissimilarity_matrix`, over the field of rational numbers.  
// /// 
// /// - Edges of weight `>= dissimilarity_max` are excluded.
// /// - Homology is computed in dimensions 0 through `homology_dimension_max`, inclusive
// /// 
// /// Returns: `( BarcodePySimplexFilteredRational, L )`, where `L[p]` is the optimized cycle for the bar with unique id number `p`.
// #[pyfunction]
// // #[args( dissimilarity_matrix = "vec![vec![]]", homology_dimension_max="0", dissimilarity_max="None", )]
// // #[text_signature = "(dissimilarity_matrix, homology_dimension_max, dissimilarity_max, /)"]
// // #[pyo3(signature = (dissimilarity_matrix=vec![vec![0.0]], homology_dimension_max=0, dissimilarity_max=None))]
// // #[pyo3(signature = (dissimilarity_matrix, homology_dimension_max, dissimilarity_max))]
// pub fn persistent_homology_vr_optimized( 
//             dissimilarity_matrix: Vec<Vec<f64>>, 
//             homology_dimension_max: isize,
//             dissimilarity_max: Option<f64>,
//             dissimilarity_min: Option<f64>,
//         ) 
//     -> ( BarcodePySimplexFilteredRational, Vec< MinimalCyclePySimplexFilteredRational > )
//     {

//     println!("###############");
 
//     // convert the dissimilarity matrix to type FilVal
//     let dissimilarity_matrix_data = dissimilarity_matrix.iter().map(|x| x.iter().cloned().map(|x| OrderedFloat(x)).collect_vec() ).collect_vec();
//     let dissimilarity_matrix_size = dissimilarity_matrix.len();
//     let dissimilarity_max = dissimilarity_max.map(|x| OrderedFloat(x));
//     let dissimilarity_min = dissimilarity_min.map_or( 
//             { 
//                 if dissimilarity_matrix_size==0 {
//                     OrderedFloat(0.0)
//                 } else { 
//                     dissimilarity_matrix_data.iter().map(|x| x.iter()).flatten().min().unwrap().clone()
//                 } 
//             },
//             |x: f64| OrderedFloat(x),            
//         );
//     let dissimilarity_matrix = Arc::new( dissimilarity_matrix_data.into_csr( dissimilarity_matrix_size, dissimilarity_matrix_size ) );
//     // define the ring operator
//     let ring_operator = FieldRationalSize::new();
//     // define the chain complex
//     let chain_complex = ChainComplexVrFiltered::new( dissimilarity_matrix, dissimilarity_matrix_size, dissimilarity_max, dissimilarity_min, ring_operator.clone() );
//     let factored = chain_complex.factor_from_arc( homology_dimension_max );

//     // get a reference to the chain complex (needed in order to create certain iterators, due to lifetime bounds)
//     let chain_complex_ref = & chain_complex;   
//     // define an interator to run over the row indices of the boundary matrix 
//     let keymaj_vec = chain_complex.cliques_in_order( homology_dimension_max );
//     let iter_keymaj = keymaj_vec.iter().cloned();    
//     // obtain a u-match factorization of the boundary matrix
//     let factored = factor_boundary_matrix(
//                 boundary_matrix, 
//                 ring_operator, 
//                 order_comparator, 
//                 row_indices
//             );

//     let umatch = new_umatchrowmajor_with_clearing(
//             chain_complex_ref, 
//             iter_keymaj.clone(), 
//             ring_operator.clone(), 
//             OrderOperatorAutoLt::new(), 
//             OrderOperatorAutoLt::new(), 
//         );      
//     // unpack the factored boundary matrix into a barcode
//     let dim_fn = |x: &SimplexFiltered<FilVal> | x.dimension() as isize;
//     let fil_fn = |x: &SimplexFiltered<FilVal> | x.filtration();    
//     let obj_fn = |x: &SimplexFiltered<FilVal> | x.filtration().into_inner();                
//     let barcode = oat_rust::algebra::chains::barcode::barcode( &umatch, iter_keymaj.clone(), dim_fn, fil_fn, true , true);
      
//     let mut optimized_cycles                =   Vec::new();
//     use indicatif::ProgressBar;
//     // let progress_bar = ProgressBar::new( barcode.bars().len() );

//     for bar in barcode.iter() {

//         // progress_bar.inc(1);

//         if bar.dimension() == 0 { continue }
        
//         let birth_column                =   bar.birth_column();
//         let optimized                   =   minimize_cycle(
//                                                     & umatch,
//                                                     iter_keymaj.clone(),
//                                                     dim_fn,
//                                                     fil_fn,
//                                                     obj_fn,
//                                                     birth_column.clone(),
//                                                 ).ok().unwrap();   
//         let cycle_old                   =   optimized.cycle_initial().iter().cloned()
//                                                 .map(|(simplex,coeff)| ( SimplexFilteredPy::new(simplex), coeff ) ).collect();
//         let cycle_min                   =   optimized.cycle_optimal().iter().cloned()
//                                                 .map(|(simplex,coeff)| ( SimplexFilteredPy::new(simplex), coeff ) ).collect();
//         let bounding_chain              =   optimized.bounding_chain().iter().cloned()
//                                                 .map(|(simplex,coeff)| ( SimplexFilteredPy::new(simplex), coeff ) ).collect();
//         let objective_old               =   optimized.objective_initial().clone();
//         let objective_min               =   optimized.objective_optimal().clone();

//         let optimized                   =   MinimalCycle::new(cycle_old, cycle_min, objective_old, objective_min, bounding_chain);
//         let optimized                   = MinimalCyclePySimplexFilteredRational::new( optimized );
//         optimized_cycles.push( optimized );
//     }
//     let barcode = BarcodePySimplexFilteredRational::new( barcode );

//     return (barcode, optimized_cycles)
// }


//  =========================================
//  PERSISTENT RELATIVE HOMOLOGY
//  =========================================



#[pyclass]
/// 
/// The factored boundary matrix for a filtered pair of Vietoris-Rips Complexes, or equivalently, the factored boundary matrix of a filtered 
/// quotient chain complex. 
/// 
/// Using this U-match, we provide methods for: 
/// - barcodes for persistent relative homology modules
/// - relative cycle representatives via basis matching
/// - python wrappers for: 
///     - sparse boundry matrices 
///     - major and minor indices of sparse boundary matrices
///     - sparse target (codomain) COMBs (containing relative boundary basis)
///     - sparse source (domain) COMBs (containing relative cycle basis)
/// 
pub struct FactoredBoundaryMatrixVrRelative {
    prh_manager: RelativeBoundaryMatrixOracleWrapper<
        // RelativeOracle
        RelativeBoundaryMatrixOracle<                                                                     
            Arc<CsMatBase<OrderedFloat<f64>, usize, Vec<usize>, Vec<usize>, Vec<OrderedFloat<f64>>>>, 
            OrderedFloat<f64>, 
            Ratio<isize>, 
            DivisionRingNative<Ratio<isize>>, 
            OrderOperatorSubComplexFiltrationSimplices<Arc<CsMatBase<OrderedFloat<f64>, usize, Vec<usize>, Vec<usize>, Vec<OrderedFloat<f64>>>>, OrderedFloat<f64>>
        >, 
        // Filtration
        OrderedFloat<f64>,
        // Coefficient                                                                                
        Ratio<isize>,      
        // RingOperator                                                                               
        DivisionRingNative<Ratio<isize>>,    
        // OrderOperatorOracleKeyMinor                                                             
        OrderOperatorFullComplexFiltrationSimplices<                                                      
            Arc<CsMatBase<OrderedFloat<f64>, usize, Vec<usize>, Vec<usize>, Vec<OrderedFloat<f64>>>>, 
            OrderedFloat<f64>
        >, 
        // OrderOperatorOracleKeyMajor
        RelativeBoundaryMatrixRowIndexOrderOperator<  
            OrderedFloat<f64>, 
            OrderOperatorFullComplexFiltrationSimplices<Arc<CsMatBase<OrderedFloat<f64>, usize, Vec<usize>, Vec<usize>, Vec<OrderedFloat<f64>>>>, OrderedFloat<f64>>, 					
            OrderOperatorSubComplexFiltrationSimplices<Arc<CsMatBase<OrderedFloat<f64>, usize, Vec<usize>, Vec<usize>, Vec<OrderedFloat<f64>>>>, OrderedFloat<f64>>
        >, 
        // OrderOperatorOracleViewMinor -- constructed when we UMatch factor the `RelativeOracle`
        OrderOperatorByKeyCutsom<
            SimplexFiltered<OrderedFloat<f64>>, 
            Ratio<isize>, 
            (SimplexFiltered<OrderedFloat<f64>>, Ratio<isize>), 
            RelativeBoundaryMatrixRowIndexOrderOperator<
                OrderedFloat<f64>, 
                OrderOperatorFullComplexFiltrationSimplices<Arc<CsMatBase<OrderedFloat<f64>, usize, Vec<usize>, Vec<usize>, Vec<OrderedFloat<f64>>>>, OrderedFloat<f64>>, 					 					OrderOperatorSubComplexFiltrationSimplices<Arc<CsMatBase<OrderedFloat<f64>, usize, Vec<usize>, Vec<usize>, Vec<OrderedFloat<f64>>>>, OrderedFloat<f64>>
            >
        >, 
        // OrderOperatorOracleViewMajor -- constructed when we UMatch factor the `RelativeOracle`
        OrderOperatorByKeyCutsom<
            SimplexFiltered<OrderedFloat<f64>>, 
            Ratio<isize>, 
            (SimplexFiltered<OrderedFloat<f64>>, Ratio<isize>), 
            OrderOperatorFullComplexFiltrationSimplices<
                Arc<CsMatBase<OrderedFloat<f64>, usize, Vec<usize>, Vec<usize>, Vec<OrderedFloat<f64>>>>, 
                OrderedFloat<f64>
            >
        >, 
        // OrderOperatorSubComplex
        OrderOperatorSubComplexFiltrationSimplices<Arc<CsMatBase<OrderedFloat<f64>, usize, Vec<usize>, Vec<usize>, Vec<OrderedFloat<f64>>>>, OrderedFloat<f64>>
    >,
    time_to_construct_and_factor_oracle: Option<Duration>
}

// CURRENTLY UNDER CONSTRUCTION!!!
// TODO: refactor this struct 
// - should be compatible with custom filtration 
// - compute the filtered comb product and export to a CSR in python 
// - send this CSR product back to Rust to compute the basis matching U-match 
// - rest of the code should stay the same, with minimal refactoring to accomodate this change! 

// Implementation and methods of `FactoredBoundaryMatrixVrRelative`
#[pymethods]
impl FactoredBoundaryMatrixVrRelative { 

    /// 
    /// Construct a pair of Vietoris-Rips complexes and factor the boundary matrix of the associated filtered, quotient chain complex over the field of rational numbers.  
    /// 
    /// # Arguments
    /// 
    /// - `dissimilarity_matrix_full_space`: a sparse dissimilarity matrix for a point cloud; missing entries will be treated as edges that never enter the filtration. 
    /// - `dissimilarity_matrix_subspace`: a sparse dissimilarity matrix for a point cloud which is a subset of the point cloud given by `dissimilarity_matrix_full_space`. 
    /// - `homology_dimension_max`: the maximum dimension for which homology is desired. 
    /// 
    /// # Returns 
    /// 
    /// - An instance of `RelativeBoundaryMatrixOracleWrapper`, which provides methods for filtered bases of relative cycles, relative boundaries and relative homology. 
    /// 
    /// # Panics
    /// 
    /// Panics if: 
    /// - Provided dissimilarity matrices are not the same size. 
    /// - There exists a structural nonzero entry in the subspace dissimilarity matrix which is not present in the full space dissimilarity matrix.
    /// - Provided dissimilarity matrices are not symmetric. 
    /// - For either the subspace or the full space, there exists an edge with filtration parameter less than the filtration parameter of its vertices.
    /// 
    /// These safety checks are performed by the constructors for `RelativeBoundaryMatrixOracle` and `ChainComplexVrFiltered`. 
    /// 
    #[new]
    pub fn new(
        py: Python<'_>,
        dissimilarity_matrix_full_space: &PyAny,
        dissimilarity_matrix_subspace: &PyAny,
        time_benchmarking: bool, 
        homology_dimension_max: Option<isize>,    
        max_dissimilarity_threshold: Option<f64>, 
        // subspace_indices_custom_filtration: Option<Vec<Option<f64>>>
    ) -> Self

    {
        // instance data for the chain complex / oracle
        let dissimilarity_matrix_full = import_sparse_matrix(py, dissimilarity_matrix_full_space).ok().unwrap();
        let dissimilarity_matrix_sub = import_sparse_matrix(py, dissimilarity_matrix_subspace).ok().unwrap();                
        let npoints_full = dissimilarity_matrix_full.rows(); 
        let npoints_sub = dissimilarity_matrix_sub.rows(); 
        let dissimilarity_min = OrderedFloat(f64::NEG_INFINITY);
        let ring_operator = FieldRationalSize::new();

        // optional time benchmarking
        let mut start: Option<Instant> = None; 
        let mut time_to_construct_and_factor_oracle: Option<Duration> = None; 
        if time_benchmarking { 
            start = Some(Instant::now()); 
        }

        // manager struct for U-Match PRH algorithm
        let subspace_order_operator = OrderOperatorSubComplexFiltrationSimplices::new(
            Arc::new(dissimilarity_matrix_sub.clone()), 
            dissimilarity_min, 
            OrderedFloat(max_dissimilarity_threshold.unwrap_or(f64::INFINITY)),
            None
        );
        let chain_complex_data = RelativeBoundaryMatrixOracle::new(
            Arc::new(dissimilarity_matrix_full), 
            Arc::new(dissimilarity_matrix_sub), 
            npoints_full, 
            npoints_sub, 
            OrderedFloat(max_dissimilarity_threshold.unwrap_or(f64::INFINITY)), 
            dissimilarity_min, 
            ring_operator, 
            subspace_order_operator,
            (homology_dimension_max.unwrap_or(1)) as usize,
        ); 
        // U-Match factor the boundary matrix of the chain complex
        let factored = chain_complex_data.factor_from_arc(); 
        let prh_manager = RelativeBoundaryMatrixOracleWrapper::new(chain_complex_data, factored);
        
        // NOTE for developers: code chunk below is deprecated as unsafe to use 
        // The option to customize `OrderOperatorSubComplexFiltrationSimplices` is not yet tested and ready for production

        // let prh_manager = match subspace_indices_custom_filtration.is_some() {
        //     // CASE: we use the standard scale filtration for the subspace 
        //     false => {
        //         // create subspace order operator 
        //         let subspace_order_operator = OrderOperatorSubComplexFiltrationSimplices::new(
        //             Arc::new(dissimilarity_matrix_sub.clone()), 
        //             dissimilarity_min, 
        //             OrderedFloat(max_dissimilarity_threshold.unwrap_or(f64::INFINITY)),
        //             None
        //         );
        //         // define the chain complex 
        //         let chain_complex_data = RelativeBoundaryMatrixOracle::new(
        //             Arc::new(dissimilarity_matrix_full), 
        //             Arc::new(dissimilarity_matrix_sub), 
        //             npoints_full, 
        //             npoints_sub, 
        //             OrderedFloat(max_dissimilarity_threshold.unwrap_or(f64::INFINITY)), 
        //             dissimilarity_min, 
        //             ring_operator, 
        //             subspace_order_operator,
        //             (homology_dimension_max.unwrap_or(1)) as usize,
        //         ); 
        //         // U-Match factor the boundary matrix of the chain complex
        //         let factored = chain_complex_data.factor_from_arc(); 
        //         RelativeBoundaryMatrixOracleWrapper::new(chain_complex_data, factored)
        //     }
        //     // CASE: we use a customized filtration for the subspace
        //     true => {
        //         // create subspace order operator 
        //         let subspace_order_operator_custom = OrderOperatorSubComplexFiltrationSimplices::new(
        //             Arc::new(dissimilarity_matrix_sub.clone()), 
        //             dissimilarity_min, 
        //             OrderedFloat(max_dissimilarity_threshold.unwrap_or(f64::INFINITY)),
        //             Some(subspace_indices_custom_filtration.unwrap().into_iter().map(|x| 
        //                 match x.is_some() { 
        //                     true => { Some(OrderedFloat(x.unwrap())) }
        //                     false => { None }
        //                 }
        //             ).collect())
        //         ); 
        //         // define the chain complex 
        //         let chain_complex_data_custom = RelativeBoundaryMatrixOracle::new(
        //             Arc::new(dissimilarity_matrix_full), 
        //             Arc::new(dissimilarity_matrix_sub), 
        //             npoints_full, 
        //             npoints_sub, 
        //             OrderedFloat(max_dissimilarity_threshold.unwrap_or(f64::INFINITY)), 
        //             dissimilarity_min, 
        //             ring_operator, 
        //             subspace_order_operator_custom,
        //             (homology_dimension_max.unwrap_or(2) + 1) as usize,
        //         ); 
        //         // U-Match factor the boundary matrix of the chain complex
        //         let factored_custom = chain_complex_data_custom.factor_from_arc(); 
        //         RelativeBoundaryMatrixOracleWrapper::new(chain_complex_data_custom, factored_custom)
        //     }   
        // }; 

        // stop timer: this accounts for elapsed time when constructing and U-match factoring the oracle! 
        if start.is_some() { 
            time_to_construct_and_factor_oracle = Some(start.unwrap().elapsed());
        }

        return FactoredBoundaryMatrixVrRelative{ prh_manager, time_to_construct_and_factor_oracle } 
    }

    /// 
    /// Returns the row indices for the boundary matrix of a filtered, quotient chain complex in sorted order.
    /// 
    /// If the max homology dimension passed by the user when factoring the boundary matrix is `d`, then the indices include
    /// 
    /// - every simplex of dimension `<= d`, and 
    /// - every simplex of dimension `d+1` that pairs with a simplex of dimension `d`.
    /// 
    pub fn row_indices_boundary_matrix(
        &self, 
    ) -> ForExport<(Vec<SimplexFiltered<OrderedFloat<f64>>>, OrderOperatorSubComplexFiltrationSimplices<Arc<CsMatBase<OrderedFloat<f64>, usize, Vec<usize>, Vec<usize>, Vec<OrderedFloat<f64>>>>, OrderedFloat<f64>>)>
    
    {
        (self.prh_manager.hash_sorter_for_row_indices_of_boundary_oracle.sorted.clone(), self.prh_manager.order_operator_sub_complex.clone()).export() 
    }      

    /// 
    /// Returns the column indices for the boundary matrix of a filtered, quotient chain complex in sorted order.
    /// 
    /// If the max homology dimension passed by the user when factoring the boundary matrix is `d`, then the indices include
    /// 
    /// - every simplex of dimension `<= d`, and 
    /// - every simplex of dimension `d+1` that pairs with a simplex of dimension `d`.
    /// 
    pub fn column_indices_boundary_matrix(
        &self
    ) -> ForExport<(Vec<SimplexFiltered<OrderedFloat<f64>>>, OrderOperatorSubComplexFiltrationSimplices<Arc<CsMatBase<OrderedFloat<f64>, usize, Vec<usize>, Vec<usize>, Vec<OrderedFloat<f64>>>>, OrderedFloat<f64>>)>
    
    { 
        (self.prh_manager.hash_sorter_for_column_indices_of_boundary_oracle.sorted.clone(), self.prh_manager.order_operator_sub_complex.clone()).export() 
    }                  

    /// 
    /// Returns the boundary matrix of a filtered, quotient chain complex formatted as a `scipy.sparse.csr_matrix` with rows/columns labeled by simplices.
    /// 
    /// This function assignes a bijection from indices to integers for both the rows and columns. Thus, the compressed, row-major 
    /// matrix which is returned is indexed by integers. However, indices/simplices can be retrieved using the lists which are 
    /// returned by `self.row_indices_boundary_matrix()` and `self.column_indices_boundary_matrix()`.
    /// 
    pub fn boundary_matrix(
        &self
    ) -> ForExport<CsMatBase<Ratio<isize>, usize, Vec<usize>, Vec<usize>, Vec<Ratio<isize>>>>

    {   
        let row_indices = self.prh_manager.hash_sorter_for_row_indices_of_boundary_oracle.sorted.clone();
        let col_indices = self.prh_manager.hash_sorter_for_column_indices_of_boundary_oracle.sorted.clone();
        let mapping = Arc::new(self.prh_manager.mapping.clone()); 
        lazy_oracle_to_csr_view_major(mapping, row_indices, col_indices).export()
    }


    /// 
    /// Returns the source (domain) COMB for the U-match of the boundary matrix of a filtered, quotient chain complex formatted as a 
    /// `scipy.sparse.csr_matrix` with rows/column labeled by simplices.
    /// 
    /// The columns of this matrix contain a basis for relative cycles. 
    /// 
    /// This function assignes a bijection from indices to (a subset of) integers for both the rows and columns. Thus, the compressed, row-major 
    /// matrix which is returned is indexed by integers. However, row (resp. column) indices/simplices can be retrieved using the lists which are 
    /// returned by `self.row_indices_boundary_matrix` (resp. `self.column_indices_boundary_matrix`).
    /// 
    pub fn comb_source(
        &self
    ) -> ForExport<CsMatBase<Ratio<isize>, usize, Vec<usize>, Vec<usize>, Vec<Ratio<isize>>>>
    
    {   
        // NOTE: source COMB has identically indexed rows and columns inherited from the column indices of the factored oracle! 
        let indices = self.prh_manager.hash_sorter_for_column_indices_of_boundary_oracle.sorted.clone();
        let mapping = self.prh_manager.umatch.comb_domain().clone(); 
        lazy_oracle_to_csr_view_major(mapping, indices.clone(), indices).export()
    }     

    /// 
    /// Returns the target (codomain) COMB for the U-match of the boundary matrix of a filtered, quotient chain complex formatted as a 
    /// `scipy.sparse.csr_matrix` with rows/column labeled by simplices.
    /// 
    /// The columns of this matrix contain a basis for relative boundaries. 
    /// 
    /// This function assignes a bijection from indices to (a subset of) integers for both the rows and columns. Thus, the compressed, row-major 
    /// matrix which is returned is indexed by integers. However, row (resp. column) indices/simplices can be retrieved using the lists which are 
    /// returned by `self.row_indices_boundary_matrix` (resp. `self.column_indices_boundary_matrix`).
    /// 
    pub fn comb_target(
        &self
    ) -> ForExport<CsMatBase<Ratio<isize>, usize, Vec<usize>, Vec<usize>, Vec<Ratio<isize>>>>
    
    {    
        // NOTE: target COMB has identically indexed rows and columns inherited from the row indices of the factored oracle! 
        let indices = self.prh_manager.hash_sorter_for_row_indices_of_boundary_oracle.sorted.clone();
        let mapping = self.prh_manager.umatch.comb_codomain().clone(); 
        lazy_oracle_to_csr_view_major(mapping, indices.clone(), indices.clone()).export()
    }  

    /// 
    /// Returns the inverse target (codomain) COMB for the U-match of the boundary matrix of a filtered, quotient chain complex formatted as a 
    /// `scipy.sparse.csr_matrix` with rows/column labeled by simplices.
    /// 
    /// This function assignes a bijection from indices to (a subset of) integers for both the rows and columns. Thus, the compressed, row-major 
    /// matrix which is returned is indexed by integers. However, row (resp. column) indices/simplices can be retrieved using the lists which are 
    /// returned by `self.row_indices_boundary_matrix` (resp. `self.column_indices_boundary_matrix`).
    /// 
    pub fn comb_target_inverse(
        &self
    ) -> ForExport<CsMatBase<Ratio<isize>, usize, Vec<usize>, Vec<usize>, Vec<Ratio<isize>>>>
    
    {    
        // NOTE: target COMB has identically indexed rows and columns inherited from the row indices of the factored oracle! 
        let indices = self.prh_manager.hash_sorter_for_row_indices_of_boundary_oracle.sorted.clone();
        let mapping = self.prh_manager.umatch.comb_codomain_inv().clone(); 
        lazy_oracle_to_csr_view_major(mapping, indices.clone(), indices).export()
    }  

    /// 
    /// Returns the index matching/matching matrix for the U-match of the boundary matrix of a filtered, quotient chain complex formatted as a 
    /// `scipy.sparse.csr_matrix` with rows/column labeled by simplices.
    /// 
    /// This function assignes a bijection from indices to (a subset of) integers for both the rows and columns. Thus, the compressed, row-major 
    /// matrix which is returned is indexed by integers. However, row (resp. column) indices/simplices can be retrieved using the lists which are 
    /// returned by `self.row_indices_boundary_matrix` (resp. `self.column_indices_boundary_matrix`).
    /// 
    pub fn index_matching(
        &self
    ) -> ForExport<CsMatBase<Ratio<isize>, usize, Vec<usize>, Vec<usize>, Vec<Ratio<isize>>>>

    {
        // NOTE: index matching (or matching matrix) directly inherits the indices of the factored oracle. 
        let row_indices = self.prh_manager.hash_sorter_for_row_indices_of_boundary_oracle.sorted.clone();
        let col_indices = self.prh_manager.hash_sorter_for_column_indices_of_boundary_oracle.sorted.clone();
        let mapping = self.prh_manager.umatch.matching_ref(); 
        lazy_oracle_to_csr_view_major(mapping, row_indices, col_indices).export()
    }
    
    // /// 
    // /// Given the inverse target COMB (left) and source COMB (right) of the U-match of a filtered, quotient chain complex, U-match 
    // /// factor their product and export the resulting Target COMB to python as a CSR matrix. More generally, if matching the the column 
    // /// span of matrices A and B, then we take `left` = A^{-1} and `right` = B. 
    // ///  
    // pub fn persistent_relative_homology_2(
    //     &self, 
    //     py: Python<'_>
    // ) // -> ForExport<CsMatBase<Ratio<isize>, usize, Vec<usize>, Vec<usize>, Vec<Ratio<isize>>>> 

    // {
    //     // this function will need ALL of the following lists of simplices, which is why `Self` owns copies of all of them! 
    //     let simplices_subcomplex_order = self.prh_manager.simplices_subcomplex_order_ref();
    //     let simplices_full_complex_order = self.prh_manager.simplices_full_complex_order_ref();
    //     let simplices_relative_cycle_order = self.prh_manager.simplices_relative_cycle_order_ref(); 
    //     let simplices_relative_boundary_order = self.prh_manager.simplices_relative_boundary_order_ref(); 
        
    //     // Step 1a: get inverse target COMB as sparse oracle indexed by simplices
    //     let comb_target_inv = self.prh_manager.umatch.comb_codomain_inv(); 
    //     let comb_target_inv_csr = lazy_oracle_to_csr_view_major(
    //         comb_target_inv, 
    //         simplices_subcomplex_order.clone(), // these lists are dropped from memory once the CSR matrix is returned
    //         simplices_subcomplex_order.clone()
    //     );
    //     // the above is an integer indexed CSR matrix, and the below wraps it with simplex indices and equips it with OAT matrix oracle traits!
    //     let left = CompressedSparse::new(
    //         comb_target_inv_csr, 
    //         simplices_subcomplex_order.clone(), 
    //         simplices_subcomplex_order.clone() 
    //     ); 

    //     // Step 1b: get source COMB as sparse oracle indexed by simplices
    //     let comb_source = self.prh_manager.umatch.comb_domain(); 
    //     let comb_source_csr = lazy_oracle_to_csr_view_major(
    //         comb_source, 
    //         simplices_full_complex_order.clone(), // these lists are dropped from memory once the CSR matrix is returned
    //         simplices_full_complex_order.clone()
    //     );
    //     let right = CompressedSparse::new(
    //         comb_source_csr, 
    //         simplices_full_complex_order.clone(), 
    //         simplices_full_complex_order.clone() 
    //     ); 

    //     // Step 2: create lazy product of the oracles
    //     let filtered_comb_product = BimajorProductMatrix::new(
    //         left, 
    //         right,
    //         DivisionRingNative::new(), 
    //         simplices_relative_boundary_order.clone(),                               // row indices 
    //         self.prh_manager.hash_sorter_for_target_comb_order_operator.to_owned(),  // order operator on row indices
    //         self.prh_manager.hash_sorter_for_source_comb_order_operator.to_owned(),  // order operator on column indices
    //     );

    //     // Step 3: U-match factor the lazy product 
    //     let comb_product = Arc::new(filtered_comb_product); 
    //     let basis_matching_umatch = Umatch::factor(
    //         comb_product.clone(),                                                          
    //         simplices_relative_boundary_order.clone().into_iter().rev(), 
    //         comb_product.ring_operator.clone(), 
    //         self.prh_manager.hash_sorter_for_target_comb_order_operator.to_owned(),
    //         self.prh_manager.hash_sorter_for_source_comb_order_operator.to_owned() 
    //     );

    //     // Step 4: get target COMB of the "basis matching" U-match (T)
    //     let comb_target_t = basis_matching_umatch.comb_codomain();  
        
    //     // Step 5: get the original target COMB (A)
    //     let comb_target_a = self.prh_manager.umatch.comb_codomain(); 

    //     // Step 6: lazy product matrix AT
    //     let matched_basis = BimajorProductMatrix::new(
    //         comb_target_a, 
    //         comb_target_t,
    //         DivisionRingNative::new(), 
    //         simplices_subcomplex_order.clone(),                                          // row indices 
    //         self.prh_manager.hash_sorter_for_row_indices_of_boundary_oracle.to_owned(),  // order operator on row indices
    //         self.prh_manager.hash_sorter_for_target_comb_order_operator.to_owned(),      // order operator on column indices
    //     );

    //     // Step 7: unpack this sparse product to a PRH module and barcode (See below)
    // }


    // ///
    // /// Returns the matched basis for the boundary matrix of a filtered, quotient chain complex. Specifically, this is a single matrix whose columns 
    // /// contain bases for all relative cycles and all relative boundaires of the underlying quotient space, allowing for simplified extraction of 
    // /// generators for relative homology classes. 
    // /// 
    // /// The exported matched basis is formatted as a `scipy.sparse.csr_matrix` with rows/column labeled by simplices.
    // /// 
    // /// This function assignes a bijection from indices to (a subset of) integers for both the rows and columns. Thus, the compressed, row-major 
    // /// matrix which is returned is indexed by integers. However, row (resp. column) indices/simplices can be retrieved using the lists which are 
    // /// returned by `self.row_indices_boundary_matrix` (resp. `self.column_indices_matched_basis`). 
    // /// 
    // pub fn matched_basis(
    //     &self
    // ) -> ForExport<CsMatBase<Ratio<isize>, usize, Vec<usize>, Vec<usize>, Vec<Ratio<isize>>>>

    // {
    //     // get data needed to export
    //     let row_indices = self.prh_manager.mapping.get_sorted_keys_major_or_minor(true);
    //     let col_indices = self.prh_manager.hash_sorter_for_source_comb_order_operator.sorted.clone();
    //     let comb_product_factored = self.prh_manager.lazy_comb_product_factored(); 
    //     let mapping = self.prh_manager.lazy_matched_basis(&comb_product_factored).0; 
    //     let shape = (row_indices.len(), col_indices.len()); 

    //     // assign a bijection from keys to integers for both the rows and columns
    //     let row_bijection: HashMap<_,_> = row_indices.iter().cloned().enumerate().map(|(x,y)| (y,x) ).collect();
    //     let col_bijection: HashMap<_,_> = col_indices.iter().cloned().enumerate().map(|(x,y)| (y,x) ).collect();

    //     // format data from the matrix to export ... collect it in the vectors below
    //     let mut indices_row = Vec::new();
    //     let mut indices_col = Vec::new();
    //     let mut vals = Vec::new();
    //     for col_index in col_indices.iter().cloned() {
    //         for (row_index, coefficient) in mapping.view_minor_descend(col_index.clone()) {
    //             indices_row.push(row_bijection[&row_index.clone()].clone()); 
    //             indices_col.push(col_bijection[&col_index.clone()].clone()); 
    //             vals.push(coefficient); 
    //         }
    //     }

    //     // export the formatted data
    //     TriMatBase::from_triplets(shape, indices_row, indices_col, vals).to_csr().export() 
    // }


    // /// 
    // /// Returns the column indices of the matched basis for the boundary matrix of a filtered, quotient chain complex along with 
    // /// the birth and (optional) death of the correspondeing column views as generators for relative homology classes. 
    // /// 
    // pub fn column_indices_matched_basis(
    //     &self, 
    // ) -> ForExport< Vec<(SimplexFiltered<OrderedFloat<f64>>, Option<OrderedFloat<f64>>, Option<OrderedFloat<f64>>)> >
    
    // { 
    //     // get the key list and sort by relative cycle birth
    //     let keys = self.prh_manager.hash_sorter_for_source_comb_order_operator.sorted.clone();
    //     // get vectors of birth and death filtrations 
    //     let relative_cycle_births = keys.iter().map( |x| self.prh_manager.source_comb_order_operator.relative_cycle_birth(x)).collect_vec(); 
    //     let relative_boundary_births = keys.iter().map( |x| self.prh_manager.target_comb_order_operator.relative_boundary_birth(x)).collect_vec(); 
    //     // zip into a single vector and export to python
    //     keys.into_iter()
    //         .zip(relative_cycle_births)
    //         .zip(relative_boundary_births)
    //         .map(|((key, birth), death)| (key, birth, death))
    //         .collect_vec()
    //         .export()
    // }

    /// 
    /// Extract a barcode and a basis of relative cycle representatives. 
    /// 
    /// Computes persistent homology given the boundary matrix of a filtered, quotient chain complex. 
    /// In other words, a filtered basis of relative cycles mod relative boundaries.  
    /// 
    /// - Edges of weight `>= dissimilarity_max` are excluded.
    /// - Relative homology is computed in dimensions 0 through `homology_dimension_max`, inclusive.
    /// 
    /// # Arguments: 
    /// 
    /// - `return_cycle_representatives` is a boolean that determines if the constructed barcode is accompanied by a persistence module, or equivalently, a basis of cycle representatives. 
    /// - `return_bounding_chains` is a boolean that determines if the bounding chain of a cycle representative is returned, when it exists. 
    /// - `trim_subcomplex_simplices_from_cycle_representatives` is a boolean that determines if cycle representatives include or suppress supspace simplices. If true, these simplices are excluded from representatives.
    /// 
    /// Note that this function WILL panic if `trim_cycle_representatives` = `true` and `return_cycle_representatives` = `false`. 
    /// 
    /// # Returns: 
    /// 
    /// A Pandas data frame. 
    /// 
    pub fn persistent_relative_homology(
        &self,       
        py: Python<'_>,
        return_cycle_representatives: bool, 
        return_bounding_chains: bool,
        trim_subcomplex_simplices_from_cycle_representatives: bool
    ) -> Py<PyAny>
    
    {   
        // safety check: 
        if trim_subcomplex_simplices_from_cycle_representatives && !return_cycle_representatives { 
            panic!(" \n\n Error: Attempted to trim relative cycle representatives without first computing them. Ensure to set `return_cycle_representatives` = `true`. \n This message is generated by OAT. \n\n"); 
        }
        
        // cycle and boundary order operators
        let source_comb_order_operator = self.prh_manager.source_comb_order_operator.clone();
        let target_comb_order_operator = self.prh_manager.target_comb_order_operator.clone();

        // source COMB from first U-match, matched basis, and target COMB from second U-match --> for cycle representatives and bounding chains
        let comb_source = self.prh_manager.umatch.comb_domain(); 
        let comb_product_factored = self.prh_manager.lazy_comb_product_factored(); 
        let (matched_basis, comb_target_of_second_umatch) = match return_cycle_representatives || return_bounding_chains { 
            true => { 
                let basis_matching = self.prh_manager.lazy_matched_basis(&comb_product_factored); 
                ( Some(basis_matching.0), Some(basis_matching.1) )
            }, 
            false => { (None, None) }
        }; 

        // a data struct to unpack the matched basis along with key information relating to its barcode 
        // NOTE: 
        // - the `matched_basis` as returned from `RelativeBoundaryMatrixOracleWrapper::matched_basis()` is rather messy
        // - the returned struct reflects that it is the product AT' where A and T' are the target COMBs of two different U-match decompositions
        // - we wrap it into an iterator of (key, birth, death, generator) pairs, which is cleaner to work with
        // - the iterator only includes keys that generate relative homology classes, and these keys are ordered by ascending birth of the classes they generate
        // - a cycle representative is only included if `return_cycle_representatives` = `true`
        let iter_matched_basis: Vec<(
            SimplexFiltered<OrderedFloat<f64>>,                               // column index of `matched_basis` whose column view generates a relative homology class
            OrderedFloat<f64>,                                                // the birth of the generator
            Option<OrderedFloat<f64>>,                                        // the death of the generator, which may be none
            Option<Vec<(SimplexFiltered<OrderedFloat<f64>>, Ratio<isize>)>>   // associated generator, which may be none
        )>; 

        // function closures which we pass to `barcode_relative_homology()`
        // `dim_fn` maps a simplex to its dimension
        let dim_fn = | x: &SimplexFiltered<FilVal> | x.dimension() as isize; 
        // `birth_column_to_death_column_fn` maps a birth simplex for a relative homology class to the simplex whose birth subsumes the class 
        let birth_column_to_death_column_fn = | x: &SimplexFiltered<FilVal> | { 
            let boundary_birth = self.prh_manager.target_comb_order_operator.relative_boundary_birth(x);
            if let Some(death_filtration) = boundary_birth { 
                let matched_minor_key_from_first_umatch = self.prh_manager.umatch.matching_ref().keymaj_to_keymin(x);
                // absolute boundary case
                if let Some(simplex_filtered) = matched_minor_key_from_first_umatch { 
                    // need to account for customized subcomplex filtration
                    if Some(death_filtration) < self.prh_manager.order_operator_sub_complex.diameter(&simplex_filtered.vertices) { 
                        Some(x.clone())
                    } else { 
                        Some(simplex_filtered)
                    }
                // relative (but not absolute) boundary case
                } else { 
                    comb_product_factored.matching_ref().keymin_to_keymaj(x)
                }
            } else {
                None
            }
        };
        // `birth_column_to_bounding_chain_fn` maps the birth simplex for a relative homology class to the bounding cycle of that classes cycle representative, when it exists. 
        // NOTE: 
        // - cycle representatives for subsumed homology classes (boundaries) are from the matrix AT'
        // - bounding chains typically are columns of S which match to an absolute boundary in T via the relation TM = DS 
        // - to account for this change of basis, bounding chains are columns of ST'
        // - why does this work? A and T are identical up to a permutation of columns!
        let birth_column_to_bounding_chain_fn = | x: &SimplexFiltered<FilVal> | { 
            let matched_minor_key_from_first_umatch = self.prh_manager.umatch.matching_ref().keymaj_to_keymin(x);
            if return_bounding_chains && matched_minor_key_from_first_umatch.is_some() { 
                let bounding_chain_matched_basis: Vec<(SimplexFiltered<OrderedFloat<f64>>, Ratio<isize>)> = vector_matrix_multiply_minor_descend_simplified(
                    &comb_target_of_second_umatch.clone().unwrap().view_minor_descend(matched_minor_key_from_first_umatch.clone().unwrap()).collect_vec(),
                    comb_source.clone(), 
                    self.prh_manager.umatch.ring_operator(), 
                    self.prh_manager.umatch.order_operator_major(),
                ).collect_vec();
                Some(bounding_chain_matched_basis)
            } else { 
                None 
            }
        };

        // `trim_chain_fn` removes subcomplex simplices from cycle representative (upon user request)
        let trim_chain_fn = | x: Vec<(SimplexFiltered<FilVal>, Ratio<isize>)> | self.prh_manager.trim_relative_chain(x);

        // we unpack the matched basis to an iterator over `(key, birth, death, cycle_representative)` tuples
        let keys = self.prh_manager.sort_source_comb_minor_keys_by_relative_cycle_birth();   
        iter_matched_basis = keys.clone().into_iter()
            .zip( keys.iter().filter_map( |x| source_comb_order_operator.relative_cycle_birth(x)).collect_vec() )
            .zip( keys.iter().map( |x| target_comb_order_operator.relative_boundary_birth(x)).collect_vec() )
            .map(| ((key, birth), death) | {
                match return_cycle_representatives { 
                    true => {                                                                 
                        if death.is_some() { 
                            ( key.clone(), birth, death, Some(matched_basis.clone().unwrap().view_minor_descend(key.clone()).collect()) ) 
                        } else { 
                            ( key.clone(), birth, death, Some(comb_source.view_minor_descend(key.clone()).collect()) )
                        }
                    }, 
                    false => { (key, birth, death, None) }
                }
            }).collect_vec(); 
        
        // unpack into a barcode
        let barcode = oat_rust::algebra::chains::barcode::barcode_relative_homology(
            iter_matched_basis.into_iter(), 
            dim_fn, 
            birth_column_to_death_column_fn, 
            birth_column_to_bounding_chain_fn,
            trim_chain_fn, 
            trim_subcomplex_simplices_from_cycle_representatives
        );

        // export to python 
        let dict = PyDict::new(py);
        dict.set_item( "id", 
            barcode.bars().iter().map(|x| x.id_number() ).collect_vec() ).ok().unwrap();
        dict.set_item( "dimension", 
            barcode.bars().iter().map(|x| x.birth_column().dimension() ).collect_vec() ).ok().unwrap();            
        dict.set_item( "birth", 
            barcode.bars().iter().map(|x| x.birth_f64() ).collect_vec() ).ok().unwrap();
        dict.set_item( "death", 
            barcode.bars().iter().map(|x| x.death_f64() ).collect_vec() ).ok().unwrap();
        dict.set_item( "birth simplex", 
            barcode.bars().iter().map(|x| x.birth_column().vertices() ).collect_vec() ).ok().unwrap();
        dict.set_item( "death simplex", 
            barcode.bars().iter().map(|x| x.death_column().clone().map(|x| x.vertices().clone() ) ).collect_vec()).ok().unwrap();   
        if return_cycle_representatives { 
            dict.set_item( "cycle representative (string)", 
                barcode.bars().iter().map(|x| x.cycle_representative().as_ref().unwrap().chain_to_string(self.prh_manager.umatch.ring_operator()).clone() ).collect_vec() ).ok().unwrap();
            dict.set_item( "cycle representative (data)", 
                barcode.bars().iter().map(|x| x.cycle_representative().as_ref().unwrap().clone().export() ).collect_vec() ).ok().unwrap();
            dict.set_item( "cycle nnz", 
                barcode.bars().iter().map(|x| x.cycle_representative().as_ref().map(|x| x.len() ) ).collect_vec() ).ok().unwrap();     
        }      
        if return_bounding_chains { 
            // here we need to check that the bounding chain exists before we unwrap and format as a string! 
            dict.set_item( "bounding chain (string)", 
                barcode.bars().iter().map(|x| { 
                    if let Some(bounding_chain) = x.bounding_chain().as_ref() { 
                        bounding_chain.chain_to_string(self.prh_manager.umatch.ring_operator()).clone()
                    } else { 
                        String::from("")
                    }
                }).collect_vec()).ok().unwrap();
            dict.set_item( "bounding chain (data)", 
                barcode.bars().iter().map(|x| x.bounding_chain().as_ref().map(|x| x.clone().export()) ).collect_vec() ).ok().unwrap();
            dict.set_item( "bounding nnz", 
                barcode.bars().iter().map(|x| x.bounding_chain().as_ref().map(|x| x.len() ) ).collect_vec() ).ok().unwrap();
        }
        let pandas = py.import("pandas").ok().unwrap();       
        let df: Py<PyAny> = pandas.call_method("DataFrame", ( dict, ), None).map(Into::into).ok().unwrap();
        df.call_method( py, "set_index", ( "id", ), None).map(Into::into).ok().unwrap()   
    } 

    /// 
    /// A wrapper function which calls `self.persistent_relative_homology()` and keeps track of time elpased to 
    /// extract a barcode and a basis of relative cycle representatives. 
    /// 
    /// # Arguments: 
    /// 
    /// - `return_cycle_representatives` is a boolean that determines if the constructed barcode is accompanied by a persistence module, or equivalently, a basis of cycle representatives. 
    /// 
    /// # Returns: 
    /// 
    /// The time required to compute persistent relative homology, in seconds, as a 64-bit float. Will return 0 in the case that the user has not specified `time_benchmarking` = `true`
    /// when initializing the oracle.
    /// 
    pub fn persistent_relative_homology_time_benchmarking(
        &self,       
        py: Python<'_>,
        return_cycle_representatives: bool, 
        return_bounding_chains: bool
    ) -> f64
    
    { 
        if self.time_to_construct_and_factor_oracle.is_some() { 
            let start = Instant::now(); 
            self.persistent_relative_homology(py, return_cycle_representatives, return_bounding_chains, false); 
            let time_to_compute_prh = start.elapsed();
            let total_time = self.time_to_construct_and_factor_oracle.unwrap().as_secs_f64() + time_to_compute_prh.as_secs_f64();
            return total_time
        } else { 
            return 0 as f64
        }
    }
}


#[cfg(test)]
mod tests {
    use pyo3::Python;
    use oat_rust::algebra::matrices::types::third_party::IntoCSR;

    


    // #[test]
    // fn test_barcode_fixed_symmetric_matrix() {

    //     use crate::clique_filtered::FactoredBoundaryMatrixVr;

    //     let dissimilarity_max = None;
    //     let homology_dimension_max = Some(1);

    //     let dissimilarity_matrix =
    //     vec![ 
    //     vec![0.0, 0.6016928528850207, 0.493811064571812, 0.7631842110599732, 0.6190969952854828, 0.32238439536052743, 0.5577776299243353, 0.7818974214708962, 0.07661198884101905, 0.4725681975471917, 0.11373899464129633, 0.42692474128277236, 0.8617605210898125, 0.6033834157784794, 0.6507666017239748, 0.6108287386340484, 0.6874754930701601, 0.5216650170561481, 0.1739545434174833, 0.3848087421417594],
    //     vec![0.6016928528850207, 0.0, 0.5092128196637472, 0.3972421208618373, 0.3046297569686842, 0.4124608436158862, 0.2806048596469476, 0.3519192500394136, 0.5956941890831011, 0.3891213477906711, 0.05217685800395466, 0.5673170383785954, 0.6154346905039156, 0.8410186822326671, 0.6106959601576187, 0.7283439354447504, 0.5496200412544044, 0.5000451211467285, 0.3798535242449169, 0.5243930541547187],
    //     vec![0.493811064571812, 0.5092128196637472, 0.0, 0.3561773339990194, 0.34386022814969286, 0.47820995353849394, 0.3358482108698321, 0.3112545444910565, 0.6769811259281259, 0.11951440156345605, 0.28557067972725503, 0.512837799856345, 0.14341566187913501, 0.19856500421639478, 0.5350631971916313, 0.7224583474471165, 0.6061450244826808, 0.9072555593504178, 0.42069193806319394, 0.6319175411184014],
    //     vec![0.7631842110599732, 0.3972421208618373, 0.3561773339990194, 0.0, 0.3487932780695402, 0.46719510926568875, 0.5104490109819306, 0.42786488797344424, 0.7260539344838907, 0.5216838489415861, 0.3665979132978837, 0.1739258675892733, 0.4606050949827942, 0.40558969160305447, 0.5658659589949734, 0.22907682914861116, 0.8173301082204779, 0.320916283257647, 0.42774123820610455, 0.8634899734150683],
    //     vec![0.6190969952854828, 0.3046297569686842, 0.34386022814969286, 0.3487932780695402, 0.0, 0.7662218806020976, 0.46151054210129994, 0.4990724633689937, 0.8707390402111069, 0.3505745678194895, 0.5189917539728075, 0.42824076055710325, 0.1842675961210688, 0.8458600272472917, 0.24228395929341928, 0.14468843668941522, 0.28900265054271523, 0.5305753485287981, 0.5462378624754798, 0.8095581217994358],
    //     vec![0.32238439536052743, 0.4124608436158862, 0.47820995353849394, 0.46719510926568875, 0.7662218806020976, 0.0, 0.3845952142742255, 0.2536554291638623, 0.6413595103144578, 0.6128904410415779, 0.5285348472099765, 0.5177751670806997, 0.711864519210214, 0.62428815517063, 0.19414566417205048, 0.7066233019294025, 0.43615946930926086, 0.6186740371220466, 0.18308366800056086, 0.6834811848495779],
    //     vec![0.5577776299243353, 0.2806048596469476, 0.3358482108698321, 0.5104490109819306, 0.46151054210129994, 0.3845952142742255, 0.0, 0.694751663136327, 0.3474705875777475, 0.26206817949657735, 0.6336863206261203, 0.26798771265418375, 0.14444456010669526, 0.6854355294928525, 0.09457649870433515, 0.7190028894889605, 0.037081784782752036, 0.37413897799597495, 0.4989135518265708, 0.3728811748113052],
    //     vec![0.7818974214708962, 0.3519192500394136, 0.3112545444910565, 0.42786488797344424, 0.4990724633689937, 0.2536554291638623, 0.694751663136327, 0.0, 0.17392906677117737, 0.6210156343133215, 0.5375749239944999, 0.5187858806627833, 0.5929340790641354, 0.7712449339329094, 0.3059336215936842, 0.36033157987432385, 0.28570096380399235, 0.04339918302952661, 0.29419322463799524, 0.2429942786113325],
    //     vec![0.07661198884101905, 0.5956941890831011, 0.6769811259281259, 0.7260539344838907, 0.8707390402111069, 0.6413595103144578, 0.3474705875777475, 0.17392906677117737, 0.0, 0.38767381994292616, 0.458781018569824, 0.4517193143860384, 0.4113984645352643, 0.21272714858166386, 0.4293977593552041, 0.6653615561279136, 0.964931953987687, 0.18254377535411093, 0.28709617555076605, 0.554288129648074],
    //     vec![0.4725681975471917, 0.3891213477906711, 0.11951440156345605, 0.5216838489415861, 0.3505745678194895, 0.6128904410415779, 0.26206817949657735, 0.6210156343133215, 0.38767381994292616, 0.0, 0.5159333989612956, 0.4175055055353978, 0.1623817553586221, 0.2509588162503712, 0.5131209051562422, 0.6430031786739608, 0.7268562340691295, 0.19940288391942473, 0.4270267130780456, 0.5342481723480923],
    //     vec![0.11373899464129633, 0.05217685800395466, 0.28557067972725503, 0.3665979132978837, 0.5189917539728075, 0.5285348472099765, 0.6336863206261203, 0.5375749239944999, 0.458781018569824, 0.5159333989612956, 0.0, 0.12132671005368023, 0.7324788222379005, 0.406730119748273, 0.45044677792578536, 0.9318540754195065, 0.4075271777861631, 0.7319995137475207, 0.15237965124911634, 0.5429616218744323],
    //     vec![0.42692474128277236, 0.5673170383785954, 0.512837799856345, 0.1739258675892733, 0.42824076055710325, 0.5177751670806997, 0.26798771265418375, 0.5187858806627833, 0.4517193143860384, 0.4175055055353978, 0.12132671005368023, 0.0, 0.49532472161425556, 0.3020632653745947, 0.6646579793145441, 0.2693091880632087, 0.386742413025264, 0.2831998326688344, 0.3599502190526389, 0.4935662425765617],
    //     vec![0.8617605210898125, 0.6154346905039156, 0.14341566187913501, 0.4606050949827942, 0.1842675961210688, 0.711864519210214, 0.14444456010669526, 0.5929340790641354, 0.4113984645352643, 0.1623817553586221, 0.7324788222379005, 0.49532472161425556, 0.0, 0.49055724427179814, 0.7323387041095746, 0.25285282889479155, 0.5228054905023033, 0.5501041781782425, 0.4691772921034907, 0.4847299731552148],
    //     vec![0.6033834157784794, 0.8410186822326671, 0.19856500421639478, 0.40558969160305447, 0.8458600272472917, 0.62428815517063, 0.6854355294928525, 0.7712449339329094, 0.21272714858166386, 0.2509588162503712, 0.406730119748273, 0.3020632653745947, 0.49055724427179814, 0.0, 0.046310411737922164, 0.48601695582724214, 0.3806904221812635, 0.6554292411367946, 0.3304760871094675, 0.4383023912725962],
    //     vec![0.6507666017239748, 0.6106959601576187, 0.5350631971916313, 0.5658659589949734, 0.24228395929341928, 0.19414566417205048, 0.09457649870433515, 0.3059336215936842, 0.4293977593552041, 0.5131209051562422, 0.45044677792578536, 0.6646579793145441, 0.7323387041095746, 0.046310411737922164, 0.0, 0.15570429291859234, 0.6035507808993115, 0.627016949499856, 0.42846636792455217, 0.8690711833626937],
    //     vec![0.6108287386340484, 0.7283439354447504, 0.7224583474471165, 0.22907682914861116, 0.14468843668941522, 0.7066233019294025, 0.7190028894889605, 0.36033157987432385, 0.6653615561279136, 0.6430031786739608, 0.9318540754195065, 0.2693091880632087, 0.25285282889479155, 0.48601695582724214, 0.15570429291859234, 0.0, 0.4771858779979862, 0.44438375123613827, 0.34983216058393884, 0.8142058135029405],
    //     vec![0.6874754930701601, 0.5496200412544044, 0.6061450244826808, 0.8173301082204779, 0.28900265054271523, 0.43615946930926086, 0.037081784782752036, 0.28570096380399235, 0.964931953987687, 0.7268562340691295, 0.4075271777861631, 0.386742413025264, 0.5228054905023033, 0.3806904221812635, 0.6035507808993115, 0.4771858779979862, 0.0, 0.48393447385845423, 0.6526221039553707, 0.17013104544474267],
    //     vec![0.5216650170561481, 0.5000451211467285, 0.9072555593504178, 0.320916283257647, 0.5305753485287981, 0.6186740371220466, 0.37413897799597495, 0.04339918302952661, 0.18254377535411093, 0.19940288391942473, 0.7319995137475207, 0.2831998326688344, 0.5501041781782425, 0.6554292411367946, 0.627016949499856, 0.44438375123613827, 0.48393447385845423, 0.0, 0.051247063916904145, 0.5188480070944168],
    //     vec![0.1739545434174833, 0.3798535242449169, 0.42069193806319394, 0.42774123820610455, 0.5462378624754798, 0.18308366800056086, 0.4989135518265708, 0.29419322463799524, 0.28709617555076605, 0.4270267130780456, 0.15237965124911634, 0.3599502190526389, 0.4691772921034907, 0.3304760871094675, 0.42846636792455217, 0.34983216058393884, 0.6526221039553707, 0.051247063916904145, 0.0, 0.707666916030988],
    //     vec![0.3848087421417594, 0.5243930541547187, 0.6319175411184014, 0.8634899734150683, 0.8095581217994358, 0.6834811848495779, 0.3728811748113052, 0.2429942786113325, 0.554288129648074, 0.5342481723480923, 0.5429616218744323, 0.4935662425765617, 0.4847299731552148, 0.4383023912725962, 0.8690711833626937, 0.8142058135029405, 0.17013104544474267, 0.5188480070944168, 0.707666916030988, 0.0],
    //     ];
    //     let dissimilarity_matrix = dissimilarity_matrix.into_csr(dissimilarity_matrix.len(),dissimilarity_matrix.len(),);

    //     Python::with_gil(|py| {
    //         let factored = FactoredBoundaryMatrixVr::new(py, dissimilarity_matrix, dissimilarity_max,homology_dimension_max);   
    //     });

        
    
    // }
 
}