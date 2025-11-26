//! 
//! Unit test coverage for the U-Match PRH algorithm. 
//! 

use crate::{algebra::{chains::relative::order::OrderOperatorFullComplexFiltrationSimplices, rings::operator_structs::{field_prime_order::PrimeOrderFieldOperator, ring_native::{DivisionRingNative, FieldRationalSize}}, vectors::operations::{Negate, Scale, Simplify, VectorOperations}}, utilities::{iterators::{general::PeekUnqualified, merge::hit::HitMerge}, order::{OrderOperatorByKeyCutsom, ReverseOrder}}}; 
use crate::algebra::chains::to_string::ChainToString; 
use crate::algebra::chains::relative::traits::{FactorFromArc, SimplexDiameter, VariableSortOracleKeys}; 
use crate::algebra::chains::relative::oracle::{RelativeBoundaryMatrixOracle, RelativeBoundaryMatrixOracleWrapper};
use crate::algebra::matrices::operations::multiply::vector_matrix_multiply_minor_descend_simplified;
use crate::algebra::matrices::query::{IndicesAndCoefficients, MatrixEntry, ViewColDescend, ViewRowAscend}; 
use crate::topology::simplicial::simplices::filtered::SimplexFiltered; 
use crate::utilities::order::JudgePartialOrder;

use itertools::Itertools;
use num::rational::Ratio;
use ordered_float::OrderedFloat;
use rand::prelude::*;

use super::order::OrderOperatorSubComplexFiltrationSimplices;

    //  ===========================================================
    //  Dissimilarity Matrix Structs for Testing
    //  ===========================================================  

    #[derive(Debug, Clone)]
    ///
    /// Implement a simple dissimilarity matrix oracle for testing the PRH algorithm. 
    /// Provided data, encode pairwise euclidean distance between these data as a 
    /// symmetric matrix. 
    /// 
    /// - `indexing_data` is a vector of tuples of 64-bit floats. These are the points which 
    /// "index" the rows and columns of the dissimilarity matrix. 
    /// - `data` is a vector of tuples of 64-bit floats. These constitute the actual data points
    /// to be recorded in the dissimilarity matrix. This field can either be identical to 
    /// 'indexing_data` or some subset of 'indexing_data`. 
    /// 
    /// # Safety Checks
    /// 
    /// The standard safety checks that would be performed to ensure proper structure of the 
    /// dissimilarity matrix are ignored here, and deferred to the constructors of the relevant 
    /// structs which are being tested by this module. 
    /// 
    struct TestDissimilarityMatrixFromData {
        indexing_data: Vec<(f64, f64)>, 
        data: Vec<(f64, f64)>,
        size: usize
    }

    impl TestDissimilarityMatrixFromData { 

        fn new(indexing_data: Vec<(f64, f64)>, data: Vec<(f64, f64)>) -> Self { 
            let size = indexing_data.iter().count();
            TestDissimilarityMatrixFromData { 
                indexing_data, 
                data, 
                size
            }
        }

        fn euclidean_distance(&self, pt1: (f64, f64), pt2: (f64, f64)) -> f64 { 
            let dx = pt2.0 - pt1.0; 
            let dy = pt2.1 - pt1.1; 
            return (dx*dx + dy*dy).sqrt();  
        }
    }

    // implement `IndicesAndCoefficients` for `DissimilarityMatrix`
    // note that rows and columns (keys) are zero indexed! 
    impl IndicesAndCoefficients for TestDissimilarityMatrixFromData {
        type EntryMajor     =   ( usize, OrderedFloat<f64> );  // (RowIndex, Coefficient)
        type EntryMinor     =   ( usize, OrderedFloat<f64> );  // (ColIndex, Coefficient)
        type RowIndex       =   usize;                         // major keys are integers
        type ColIndex       =   usize;                         // minor keys are integers
        type Coefficient    =   OrderedFloat<f64>;             // coefficients are 64 bit floats, the distance between points at `EntryMajor` and `EntryMinor`
    }

    // implement `MatrixEntry` for `DissimilarityMatrix` 
    impl MatrixEntry for TestDissimilarityMatrixFromData {
        fn entry_major_at_minor(&self, keymaj: Self::RowIndex, keymin: Self::ColIndex) -> Option< Self::Coefficient > {
            let npoints = self.data.iter().count(); 
            // CASE: constructing dissimilarity matrix entry for full space data  
            if self.size == npoints { 
                let pt1 = self.data[keymaj]; 
                let pt2 = self.data[keymin]; 
                return Some(OrderedFloat(self.euclidean_distance(pt1, pt2))); 
            }
            // CASE: constructing dissimilarity matrix entry for subspace data 
            else if self.size > npoints { 
                let pt1 = self.indexing_data[keymaj]; 
                let pt2 = self.indexing_data[keymin]; 
                if self.data.contains(&pt1) && self.data.contains(&pt2) { 
                    return Some(OrderedFloat(self.euclidean_distance(pt1, pt2))); 
                }
                else { 
                    return None; 
                }
            }
            // CASE: invalid data provided, all entries are -1.0
            else { 
                return Some(OrderedFloat(-1.0)); 
            }
        }
    }

    // implement `ViewRowAscend` for `DissimilarityMatrix`
    impl ViewRowAscend for TestDissimilarityMatrixFromData { 
        type ViewMajorAscend = Vec<Self::EntryMinor>; 
        type ViewMajorAscendIntoIter = std::vec::IntoIter<Self::EntryMajor>;
        fn view_major_ascend(&self, index: Self::RowIndex) -> Self::ViewMajorAscend {
            let mut major_view: Self::ViewMajorAscend = Vec::new(); 
            for j in 0..self.size { 
                let entry = self.entry_major_at_minor(index, j); 
                if entry.is_some() { 
                    major_view.push( (j, entry.unwrap()) ); 
                }
            }
            return major_view; 
        }
    }

    #[derive(Debug, Clone)]
    ///
    /// Implement a simple dissimilarity matrix oracle for testing the PRH algorithm. 
    /// Given a matrix, implement the necessary traits of the dissimilarity matrix. 
    /// 
    /// - `matrix_data` is a vector of vectors, where each interior vector contains 
    /// tuples of floats. Each interior vector can be seen as a row of a matrix. 
    /// 
    /// Note: this struct provides more flexibility for testing as opposed to the 
    /// above dissimilarity matrix implementation. While the above is constructed 
    /// from data by computing pairwise euclidean distances, this struct simply 
    /// wraps a provided matrix. Thus, this struct is useful for testing cases 
    /// where we expect an error or panic to occur. 
    /// 
    /// # Safety Checks
    /// 
    /// The standard safety checks that would be performed to ensure proper structure of the 
    /// dissimilarity matrix are ignored here, and deferred to the constructors of the relevant 
    /// structs which are being tested by this module. The only check which does occur is that 
    /// the provided matrix is square. 
    /// 
    struct TestDissimilarityMatrixFromMatrix {
        matrix_data: Vec< Vec< Option<f64> > >, 
        size: usize
    }

    impl TestDissimilarityMatrixFromMatrix { 
        fn new(matrix_data: Vec< Vec< Option<f64> > >) -> Self { 
            let size = matrix_data.len(); 
            for row in matrix_data.clone() { 
                if row.len() != size { 
                    panic!("Cannot construct test dissimilarity matrix from the given matrix, which is not square!"); 
                }
            }
            TestDissimilarityMatrixFromMatrix { 
                matrix_data, 
                size 
            }
        } 
    }

    // implement `IndicesAndCoefficients` for `DissimilarityMatrix`
    // note that rows and columns (keys) are zero indexed! 
    impl IndicesAndCoefficients for TestDissimilarityMatrixFromMatrix {
        type EntryMajor     =   ( usize, OrderedFloat<f64> );  // (RowIndex, Coefficient)
        type EntryMinor     =   ( usize, OrderedFloat<f64> );  // (ColIndex, Coefficient)
        type RowIndex       =   usize;                         // major keys are integers
        type ColIndex       =   usize;                         // minor keys are integers
        type Coefficient    =   OrderedFloat<f64>;             // coefficients are 64 bit floats
    }

    // implement `MatrixEntry` for `DissimilarityMatrix` 
    impl MatrixEntry for TestDissimilarityMatrixFromMatrix {
        fn entry_major_at_minor(&self, keymaj: Self::RowIndex, keymin: Self::ColIndex) -> Option< Self::Coefficient > {
            let entry = self.matrix_data[keymaj][keymin];
            if entry.is_some() { 
                return Some(OrderedFloat(entry.unwrap())); 
            } 
            else { 
                return None; 
            }
        }
    }
    
    // implement `ViewRowAscend` for `DissimilarityMatrix`
    impl ViewRowAscend for TestDissimilarityMatrixFromMatrix { 
        type ViewMajorAscend = Vec<Self::EntryMinor>; 
        type ViewMajorAscendIntoIter = std::vec::IntoIter<Self::EntryMajor>;
        fn view_major_ascend(&self, index: Self::RowIndex) -> Self::ViewMajorAscend {
            let mut major_view: Self::ViewMajorAscend = Vec::new(); 
            for j in 0..self.size { 
                let entry = self.entry_major_at_minor(index, j); 
                if entry.is_some() { 
                    major_view.push( (j, entry.unwrap()) ); 
                }
            }
            return major_view; 
        }
    }

    //  ===========================================================
    //  Generating Test Data
    //  ===========================================================  
    
    ///
    /// Generate test data points: 
    /// 
    /// - Points a through h are on the circle of radius 5, centered at (0, 0). The alphabetical order in which
    /// they are assigned is CW around the circle by increments of pi/4, starting at pi/2.
    /// - points i and j are interior to this circle, respectively given by (1,1), and (-1,-1). 
    /// 
    /// A tuple of data vectors is returned, the second being a subset of the first. 
    /// 
    /// The boolean parameters `four_points` and `remove_interior` determine the tuple that is returned. These 
    /// parameters should NEVER both be true. 
    /// 
    /// If `four_points = true`, we return: 
    /// 
    /// - {a,b,c,d,e,f,g,h,i,j}
    /// - {b,f,i,j}
    /// 
    /// If `remove_interior= true`, we return:
    /// 
    /// - {a,b,c,d,e,f,g,h}
    /// - {b,f} 
    /// 
    /// If neither are true, we return: 
    /// 
    /// - {a,b,c,d,e,f,g,h,i,j}
    /// - {i,j} 
    /// 
    fn generate_circle_test_data(four_points: bool, remove_interior: bool) -> ( Vec<(f64, f64)>, Vec<(f64, f64)> ) { 
        // Note: we define `two` here explicitly ... 
        // This is to avoid truncation errors with `num::integer::sqrt`, especially since sqrt(2) is irrational! 
        let two: f64 = 2.0;

        let a = ( 0.0, 5.0 ); 
        let b = ( ((5.0 / two.sqrt())) as f64, ((5.0 / two.sqrt())) as f64 ); 
        let c = ( 5.0, 0.0 );
        let d = ( ((5.0 / two.sqrt())) as f64, (-(5.0 / two.sqrt())) as f64 ); 
        let e = ( 0.0, -5.0 ); 
        let f = ( (-(5.0 / two.sqrt())) as f64, (-(5.0 / two.sqrt())) as f64 );
        let g = ( -5.0, 0.0 ); 
        let h = ( (-(5.0 / two.sqrt())) as f64, ((5.0 / two.sqrt())) as f64 );
        let mut points: Vec<(f64, f64)> = Vec::new(); 
        let mut points_subset: Vec<(f64, f64)> = Vec::new();
        points.extend(vec![a,b,c,d,e,f,g,h]); 

        // we only want the points on the circle
        if remove_interior { 
            points_subset.extend(vec![b, f]); 
        } 
        // we want to inlcude the interior points 
        else { 
            let i = ( 1.0, 1.0 ); 
            let j = (-1.0, -1.0 ); 
            points.extend(vec![i,j]); 
            if four_points { 
                points_subset.extend(vec![b, f, i,j]);
            } else { 
                points_subset.extend(vec![i,j]);
            }
        }
        return (points, points_subset); 
    }

    ///
    /// Generate test data points: 
    /// 
    /// Two circles of radius 2. One is centered at (-3,0) and the other is centered at (3,0). Each is circle is given 
    /// by points at integer multiples of pi/2. 
    /// 
    /// The circle on the left --centered at (-3,0)-- is labeled CW by letters a ... d, starting at pi/2. 
    /// The right circle is labaled similarly by letters e ... h. 
    /// 
    /// We return a tuple of two data vectors, one being a subset of the other. 
    /// 
    /// The subset points are given by {a, ..., d}. 
    /// 
    fn generate_side_by_side_circles_test_data() -> ( Vec<(f64, f64)>, Vec<(f64, f64)> ) { 
        let mut points: Vec<(f64, f64)> = Vec::new(); 
        let mut points_subset: Vec<(f64, f64)> = Vec::new();

        // left circle
        let a = (-3.0, 2.0);
        let b = (-1.0, 0.0);
        let c = (-3.0, -2.0);
        let d = (-5.0, 0.0);
        // right circle
        let e = (3.0, 2.0);
        let f = (5.0, 0.0);
        let g = (3.0, -2.0);
        let h = (1.0, 0.0);

        points.extend(vec![a,b,c,d,e,f,g,h]); 
        points_subset.extend(vec![a,b,c,d]); 

        return (points, points_subset);  
    }

    ///
    /// Generate test data points: 
    /// 
    /// Three points appearing on the real line. They are given by: 
    /// 
    /// - a = (-3, 0)
    /// - b = (-1,0)
    /// - c = (1, 0)
    /// - d = (3, 0)
    /// 
    /// We return a tuple of two data vectors, one being a subset of the other. 
    /// 
    /// The subset points are given by {a, d}. 
    /// 
    fn generate_line_test_data() -> ( Vec<(f64, f64)>, Vec<(f64, f64)> ) { 
        let mut points: Vec<(f64, f64)> = Vec::new(); 
        let mut points_subset: Vec<(f64, f64)> = Vec::new();

        let a = (-3.0, 0.0);
        let b = (-1.0, 0.0);
        let c = (1.0, 0.0);
        let d = (3.0, 0.0);

        points.extend(vec![a,b,c,d]); 
        points_subset.extend(vec![a,d]);

        return (points, points_subset);  
    }
    
    ///
    /// Generate test data points: 
    /// 
    /// The points of the full set of data has cardinality `num_points` and are randomly generated on a disk of given radius 
    /// centered at the origin of the plane. 
    /// 
    fn generate_random_test_data_on_disk_at_origin(radius: f64, num_points: usize, num_subset_points: usize) -> ( Vec<(f64, f64)>, Vec<(f64, f64)> ) { 
        let mut points: Vec<(f64, f64)> = Vec::new(); 
        let mut points_subset: Vec<(f64, f64)> = Vec::new();
        for i in 0..num_points { 
            let mut rng = rand::thread_rng();
            // randomly generate x coordinate
            let px: f64 = rng.gen();
            let x: f64 = px*(2.0*radius) - radius; 
            // randomly generate y coordinate
            let py: f64 = rng.gen();
            let y: f64 = py*(2.0*radius) - radius; 
            // add points to list(s)
            let point = (x,y); 
            points.push(point); 
            if i < num_subset_points { 
                points_subset.push(point); 
            }
        }
        return (points, points_subset); 
    }

    ///
    /// Generate dissimilarity matrices from data: 
    /// 
    /// Two dissimilarity matrices are returned, one for the full set of data points and one for the subset. 
    /// 
    fn generate_dissimilarity_matrices(
        points: Vec<(f64, f64)>, 
        points_subset: Vec<(f64, f64)>
    ) -> (TestDissimilarityMatrixFromData, TestDissimilarityMatrixFromData)
    
    { 
        return(
            TestDissimilarityMatrixFromData::new(points.clone(), points.clone()),
            TestDissimilarityMatrixFromData::new(points.clone(), points_subset.clone())
        ); 
    }

    ///
    /// Generate symmetric matrices: Two 5 x 5 matrices are returned. 
    /// 
    /// Note this function should only be used when testing the constructor for `RelativeBoundaryMatrixOracle`.
    /// 
    fn generate_symmetric_matrices() -> (Vec<Vec<Option<f64>>>, Vec<Vec<Option<f64>>>) { 
        // full set of data is five points
        let full_space_matrix = vec![vec![Some(0.0), Some(1.0), Some(2.5), Some(0.7), Some(3.2)], 
                                                        vec![Some(1.0), Some(0.0), Some(11.1), Some(3.0), Some(4.6)], 
                                                        vec![Some(2.5), Some(11.1), Some(0.0), Some(5.0), Some(0.4)], 
                                                        vec![Some(0.7), Some(3.0), Some(5.0), Some(0.0), Some(1.0)], 
                                                        vec![Some(3.2), Some(4.6), Some(0.4), Some(1.0), Some(0.0)]]; 
        // subset of data has one single point removed 
        let subspace_matrix = vec![vec![Some(0.0), Some(1.0), Some(2.5), None, Some(3.2)], 
                                                        vec![Some(1.0), Some(0.0), Some(11.1), None, Some(4.6)], 
                                                        vec![Some(2.5), Some(11.1), Some(0.0), None, Some(0.4)], 
                                                        vec![None, None, None, None, None], 
                                                        vec![Some(3.2), Some(4.6), Some(0.4), None, Some(0.0)]]; 
         return (full_space_matrix, subspace_matrix); 
    }

    //  ===========================================================
    //  Wrapper Functions 
    //  =========================================================== 

    fn create_oracle_from_data(
        diss_matrix_full: TestDissimilarityMatrixFromData, 
        diss_matrix_sub: TestDissimilarityMatrixFromData, 
        diss_max: OrderedFloat<f64>, 
        custom_subcomplex_data: Option<Vec<Option<OrderedFloat<f64>>>>
    ) -> RelativeBoundaryMatrixOracle<
            TestDissimilarityMatrixFromData, OrderedFloat<f64>, Ratio<isize>, DivisionRingNative<Ratio<isize>>, 
            OrderOperatorSubComplexFiltrationSimplices<TestDissimilarityMatrixFromData, OrderedFloat<f64>>
        >
    
    {
        let dissimilarity_min = OrderedFloat(f64::NEG_INFINITY); 
        let subspace_order_operator = OrderOperatorSubComplexFiltrationSimplices::new(
            diss_matrix_sub.clone(), 
            dissimilarity_min.clone(), 
            diss_max.clone(), 
            custom_subcomplex_data
        );
        RelativeBoundaryMatrixOracle::new(
            diss_matrix_full.clone(), 
            diss_matrix_sub.clone(), 
            diss_matrix_full.size, 
            diss_matrix_sub.size, 
            diss_max, 
            dissimilarity_min, 
            FieldRationalSize::new(), 
            subspace_order_operator, 
            3 as usize
        )
    }

    fn create_oracle_from_matrix(
        diss_matrix_full: TestDissimilarityMatrixFromMatrix, 
        diss_matrix_sub: TestDissimilarityMatrixFromMatrix, 
        diss_max: OrderedFloat<f64>
    ) -> RelativeBoundaryMatrixOracle<
            TestDissimilarityMatrixFromMatrix, OrderedFloat<f64>, usize, PrimeOrderFieldOperator,
            OrderOperatorSubComplexFiltrationSimplices<TestDissimilarityMatrixFromMatrix, OrderedFloat<f64>>
        >
    
    { 
        let dissimilarity_min = OrderedFloat(0.0); 
        let subspace_order_operator = OrderOperatorSubComplexFiltrationSimplices::new(
            diss_matrix_sub.clone(), 
            dissimilarity_min.clone(), 
            diss_max.clone(),
            None
        );
        RelativeBoundaryMatrixOracle::new(
            diss_matrix_full.clone(), 
            diss_matrix_sub.clone(), 
            diss_matrix_full.size, 
            diss_matrix_sub.size, 
            diss_max, 
            dissimilarity_min, 
            PrimeOrderFieldOperator::new(2), 
            subspace_order_operator,
            2 as usize
        )
    }

    //  ===========================================================
    //  Test Constructing `RelativeBoundaryMatrixOracle`
    //  ===========================================================  

    #[test]
    #[should_panic]
    /// 
    /// Case: Dissimilarity matrices are not the same size.
    /// 
    /// Require: Constructor for `RelativeBoundaryMatrixOracle` panics.
    ///  
    fn test_create_oracle_dissimilarity_matrix_size () { 
        let test_data = generate_circle_test_data(false, false); 
        let mut dissimilarity_matrices = generate_dissimilarity_matrices( 
            test_data.0, 
            test_data.1
        ); 
        // intentionally update size of the subspace matrix to be incorrect
        dissimilarity_matrices.1.size = dissimilarity_matrices.1.size - 1; 
        create_oracle_from_data(
            dissimilarity_matrices.0, 
            dissimilarity_matrices.1, 
            OrderedFloat(5.0), 
            None
        ); 
    }

    #[test]
    #[should_panic]
    /// 
    /// Case: Dissimilarity matrices do not satisfy the appropriate subspace 
    /// relationship due to incorrect entry in the subspace matrix. 
    ///  
    /// Require: Constructor for `RelativeBoundaryMatrixOracle` panics. 
    /// 
    fn test_create_oracle_subspace_dissimilarity_matrix_has_wrong_entry() { 
        let mut matrices = generate_symmetric_matrices(); 
        // intentionally update an entry of the subspace matrix to be incorrect
        matrices.1[0][2] = Some(3.14); 
        let data_full_space = TestDissimilarityMatrixFromMatrix::new(matrices.0); 
        let data_subspace = TestDissimilarityMatrixFromMatrix::new(matrices.1); 
        create_oracle_from_matrix(
            data_full_space.clone(), 
            data_subspace.clone(), 
            OrderedFloat(5.0)
        ); 
    }

    #[test]
    #[should_panic]
    /// 
    /// Case: Dissimilarity matrices do not satisfy the appropriate subspace 
    /// relationship due to incorrect entry in the full space matrix. 
    ///  
    /// Require: Constructor for `RelativeBoundaryMatrixOracle` panics. 
    /// 
    fn test_create_oracle_full_space_dissimilarity_matrix_has_wrong_entry() { 
        let mut matrices = generate_symmetric_matrices(); 
        // intentionally update an entry of the full space matrix to be incorrect
        matrices.0[0][2] = Some(3.14); 
        let data_full_space = TestDissimilarityMatrixFromMatrix::new(matrices.0); 
        let data_subspace = TestDissimilarityMatrixFromMatrix::new(matrices.1);
        create_oracle_from_matrix(
            data_full_space.clone(), 
            data_subspace.clone(), 
            OrderedFloat(5.0)
        ); 
    }

    #[test]
    #[should_panic]
    /// 
    /// Case: Dissimilarity matrices do not satisfy the appropriate subspace 
    /// relationship due to `None` entry in the full space matrix. 
    ///  
    /// Require: Constructor for `RelativeBoundaryMatrixOracle` panics. 
    /// 
    fn test_create_oracle_full_space_dissimilarity_matrix_has_none_where_subspace_dissimilarity_matrix_has_some() { 
        let mut matrices = generate_symmetric_matrices(); 
        // intentionally update an entry of the full space matrix to be incorrect
        matrices.0[0][2] = None; 
        let data_full_space = TestDissimilarityMatrixFromMatrix::new(matrices.0); 
        let data_subspace = TestDissimilarityMatrixFromMatrix::new(matrices.1); 
        create_oracle_from_matrix(
            data_full_space.clone(), 
            data_subspace.clone(), 
            OrderedFloat(5.0)
        ); 
    }

    #[test]
    /// 
    /// Case: User provides correctly structured data to the `RelativeBoundaryMatrixOracle` constructor. 
    ///  
    /// Require: Constructor for `RelativeBoundaryMatrixOracle` does not panic. 
    /// 
    fn test_create_oracle_correct() { 
        let test_data = generate_circle_test_data(false, false); 
        let dissimilarity_matrices = generate_dissimilarity_matrices( 
            test_data.0, 
            test_data.1
        ); 
        let _oracle = create_oracle_from_data(
            dissimilarity_matrices.0.clone(), 
            dissimilarity_matrices.1.clone(), 
            OrderedFloat(5.0), 
            None
        ); 
    }

    //  ===========================================================
    //  Test `RelativeBoundaryMatrixRowIndexOrderOperator`
    //  
    //  NOTE: the following structs are used to define the 
    //  functionality of this order operator: 
    //  
    //  - `OrderOperatorFullComplexFiltrationSimplices`
    //  - `OrderOperatorSubComplexFiltrationSimplices`
    //  
    //  Therefore, these cases act as tests for those structs as well. 
    //  ===========================================================  

    #[test]
    /// 
    /// Case: one simplex in X0 and one simplex in X \ X0. 
    /// 
    /// Require: simplex in X0 appears before simplex in X \ X0. 
    /// 
    fn test_one_simplex_in_each_space() { 
        let test_data = generate_circle_test_data(false, false); 
        let dissimilarity_matrices = generate_dissimilarity_matrices( 
            test_data.0, 
            test_data.1
        ); 
        // Let max dissimilarity be 2*sqrt(2). 
        // Since we are using "generate_circle_test_data" with subset data {i, j}, then at this threshold: 
        //      - the full space has points {a, ..., j} and edge [ij]
        //      - the subspace has points {i,j} and edge [ij]
        let two: f64 = 2.0; 
        let threshold = two * two.sqrt(); 
        let oracle = create_oracle_from_data(
            dissimilarity_matrices.0.clone(), 
            dissimilarity_matrices.1.clone(), 
            OrderedFloat(threshold), 
            None
        ); 
        let simplices = oracle.get_key_list();
        let len = simplices.iter().count();  
        let not_subspace_simps = &simplices[0..len-3];
        let subspace_simps = &simplices[len-3..len]; 
        let order_operator = oracle.order_operator_key_major_ref(); 
        for not_sub in not_subspace_simps { 
            assert!(!order_operator.is_subspace_simplex(not_sub)); 
        }
        for sub in subspace_simps { 
            assert!(order_operator.is_subspace_simplex(sub)); 
        }
        // assert that the relative order operator works correctly: any pairs (lhs, rhs) where lhs in X0 and rhs in X \ X0 should place lhs before rhs
        for not_sub in not_subspace_simps { 
            for sub in subspace_simps { 
                assert!(order_operator.judge_lt(sub, not_sub)); 
            }
        }
    }

    #[test]
    /// 
    /// Case: two simplices in X0 with the different diameter. 
    /// 
    /// Require: simplex with smaller diameter appears first. 
    /// 
    fn test_two_simplices_in_subspace_diameter_neq() { 
        let test_data = generate_circle_test_data(false, false); 
        let dissimilarity_matrices = generate_dissimilarity_matrices( 
            test_data.0, 
            test_data.1
        ); 
        // Let max dissimilarity be 2*sqrt(2). 
        // Since we are using "generate_circle_test_data" with subset data {i, j}, then at this threshold the subspace contains: 
        //      - the points {i,j} 
        //      - the edge [ij]
        let two: f64 = 2.0; 
        let threshold = two * two.sqrt(); 
        let oracle = create_oracle_from_data(
            dissimilarity_matrices.0.clone(), 
            dissimilarity_matrices.1.clone(), 
            OrderedFloat(threshold), 
            None
        );
        let simplices = oracle.get_key_list();
        let len = simplices.iter().count();   
        let points = &simplices[len-3..len-1];
        let simp_ij = &simplices[len-1]; 
        let order_operator = oracle.order_operator_key_major_ref();
        for point in points { 
            assert!(order_operator.is_subspace_simplex(point)); 
        }
        assert!(order_operator.is_subspace_simplex(simp_ij)); 
        // assert that the relative order operator works correctly: both points before simplex [ij]  
        for point in points { 
            assert!(order_operator.judge_lt(point, simp_ij)); 
        }
    }

    #[test]
    /// 
    /// Case: two simplices in X \ X0 with the different diameter. 
    /// 
    /// Require: simplex with smaller diameter appears first. 
    /// 
    fn test_two_simplices_in_full_space_diameter_neq() {
        let test_data = generate_circle_test_data(false, false); 
        let dissimilarity_matrices = generate_dissimilarity_matrices( 
            test_data.0, 
            test_data.1
        ); 
        // Let max dissimilarity be 3.7
        // Since we are using "generate_circle_test_data", then at this threshold the full space contains: 
        //      - the points {a, ..., j}
        //      - the edges {[ij], [bi], [fj]}
        let threshold = 3.7; 
        let oracle = create_oracle_from_data(
            dissimilarity_matrices.0.clone(), 
            dissimilarity_matrices.1.clone(), 
            OrderedFloat(threshold), 
            None
        ); 
        let simplices = oracle.get_key_list();
        let len = simplices.iter().count(); 
        let points = &simplices[0..len-5]; 
        let edges = &simplices[len-3..len-1]; 
        let order_operator = oracle.order_operator_key_major_ref(); 
        for point in points { 
            assert!(!order_operator.is_subspace_simplex(point)); 
        }
        for edge in edges { 
            assert!(!order_operator.is_subspace_simplex(edge)); 
        }
        // assert that the relative order operator works correctly: for any x in `points` and y in `edges`, the order operator gives F(x) < F(y) 
        for x in points { 
            for y in edges { 
                assert!(order_operator.judge_lt(x, y));
            }
        }
    }

    #[test]
    /// 
    /// Case: two simplices in X0 with same diameter. 
    /// 
    /// Require: simplices are ordered lexicographically. 
    /// 
    fn test_two_simplices_in_subspace_diameter_eq() { 
        let test_data = generate_circle_test_data(true, false); 
        let dissimilarity_matrices = generate_dissimilarity_matrices( 
            test_data.0, 
            test_data.1
        ); 
        // Let max dissimilarity be 3.7
        // Since we are using "generate_circle_test_data" with subset data {b, f, i, j}, then at this threshold the subspace contains: 
        //      - the points {b, f, i, j}
        //      - the edges {[ij], [bi], [fj]} ... where [bi] and [fj] have identical diameter/filtration
        let threshold = 3.7; 
        let oracle = create_oracle_from_data(
            dissimilarity_matrices.0.clone(), 
            dissimilarity_matrices.1.clone(), 
            OrderedFloat(threshold), 
            None
        ); 
        let simplices = oracle.get_key_list();
        let len = simplices.iter().count(); 
        let simp_bi = simplices[len-3].clone(); 
        let simp_fj = simplices[len-2].clone(); 
        assert!(oracle.row_index_order_operator.is_subspace_simplex(&simp_bi));
        assert!(oracle.row_index_order_operator.is_subspace_simplex(&simp_fj)); 
        // assert that the relative order operator works correctly: simplices ordered lexicographically
        assert!(oracle.row_index_order_operator.judge_lt(&simp_bi, &simp_fj)); 
    }

    #[test]
    /// 
    /// Case: two simplices in X \ X0 with same diameter. 
    /// 
    /// Require: simpices are ordered lexicographically. 
    /// 
    fn test_two_simplices_in_full_space_diameter_eq() { 
        let test_data = generate_circle_test_data(false, false); 
        let dissimilarity_matrices = generate_dissimilarity_matrices( 
            test_data.0, 
            test_data.1
        ); 
        // Let max dissimilarity be 3.7
        // Since we are using "generate_circle_test_data", then at this threshold the full space contains: 
        //      - the points {a, ..., j}
        //      - the edges {[ij], [bi], [fj]}
        let threshold = 3.7;  
        let oracle = create_oracle_from_data(
            dissimilarity_matrices.0.clone(), 
            dissimilarity_matrices.1.clone(), 
            OrderedFloat(threshold), 
            None
        ); 
        let simplices = oracle.get_key_list();
        let len = simplices.iter().count(); 
        let simp_bi = simplices[len-3].clone(); 
        let simp_fj = simplices[len-2].clone(); 
        assert_eq!(oracle.chain_complex_vr_full_space.diameter(&simp_bi.vertices), oracle.chain_complex_vr_full_space.diameter(&simp_fj.vertices));
        assert!(!oracle.row_index_order_operator.is_subspace_simplex(&simp_bi));
        assert!(!oracle.row_index_order_operator.is_subspace_simplex(&simp_fj)); 
        // assert that the relative order operator works correctly: simplices ordered lexicographically
        assert!(oracle.row_index_order_operator.judge_lt(&simp_bi, &simp_fj)); 
    }

    #[test]
    /// 
    /// Case: two simplices in X0 with identical filtration in full complex and different filtration in subcomplex. 
    /// 
    /// Require: simpices are ordered lexicographically by F and by diameter in G. 
    /// 
    fn test_subcomplex_simplex_custom_diameter() { 
        let test_data = generate_circle_test_data(true, false); 
        let dissimilarity_matrices = generate_dissimilarity_matrices( 
            test_data.0, 
            test_data.1
        ); 
        // Let max dissimilarity be 3.7
        // Since we are using "generate_circle_test_data", then at this threshold the full space contains: 
        //      - the points {a, ..., j}
        //      - the edges {[ij], [bi], [fj]}
        // and the subspace contains 
        //      - the points {b, f, i, j}
        //      - the edges {[ij], [bi], [fj]}
        let threshold = 3.7;  
        // Customize the subcomplex filtration such that F([bi]) < F([fj]) but G([fj]) < G([bi]) 
        let custom_subcomplex_data = Some(vec![
            None, 
            Some(OrderedFloat(5.0)), // 0-simplex [b]
            None, 
            None, 
            None, 
            None, 
            None, 
            None, 
            Some(OrderedFloat(5.0)), // 0-simplex [i]
            None
        ]); 
        // Now we create the oracle
        let oracle = create_oracle_from_data(
            dissimilarity_matrices.0.clone(), 
            dissimilarity_matrices.1.clone(), 
            OrderedFloat(threshold), 
            custom_subcomplex_data
        ); 
        let simplices = oracle.get_key_list();
        let len = simplices.iter().count(); 
        let simp_bi = simplices[len-3].clone(); 
        let simp_fj = simplices[len-2].clone(); 
        // assert that [bi] and [fj] have same diameter, and are ordered lexicographically by F
        assert_eq!(oracle.chain_complex_vr_full_space.diameter(&simp_bi.vertices), oracle.chain_complex_vr_full_space.diameter(&simp_fj.vertices));
        assert!(oracle.column_index_order_operator.judge_lt(&simp_bi, &simp_fj));
        // assert that [bi] and [fj] are both members of X0
        assert!(oracle.row_index_order_operator.is_subspace_simplex(&simp_bi));
        assert!(oracle.row_index_order_operator.is_subspace_simplex(&simp_fj)); 
        // assert that G([fj]) < G([bi]) with diameter of [bi] being 5.0
        assert!(oracle.subspace_filtration_order_operator.judge_lt(&simp_fj, &simp_bi)); 
        assert_eq!(oracle.subspace_filtration_order_operator.diameter(&simp_bi.vertices).unwrap().into_inner(), 5.0);
    }

    //  ===========================================================
    //  Test relative cycle/boundary order operators
    //  ===========================================================

    #[test]
    /// 
    /// Ensure that `SourceColumnIndexOrderOperator`: 
    /// 
    /// - correctly identifies relative cycles 
    /// - correctly compares relative cycles with different filtration 
    /// - lexicographically compares relative cycles with equivalent filtration
    /// 
    fn test_relative_cycle_birth() { 
        let test_data = generate_circle_test_data(true, false); 
        let dissimilarity_matrices = generate_dissimilarity_matrices( 
            test_data.0, 
            test_data.1
        ); 
        // let max dissimilarity be 3.9
        // since we are using "generate_circle_test_data" with subset data {b, f, i, j}, then at this threshold the full space contains: 
        //      - the points {a, ..., j}
        //      - the edges {[ab], [bc], [cd], ..., [gh], [ha], [bi], [fj], [ij]}
        // and the subspace contians: 
        //      - the points {b, f, i, j}
        //      - the edges {[bi], [fj], [ij]}
        let subspace_order_operator = OrderOperatorSubComplexFiltrationSimplices::new(
            dissimilarity_matrices.1.clone(), 
            OrderedFloat(0.0), 
            OrderedFloat(3.9),
            None
        );
        let oracle = RelativeBoundaryMatrixOracle::new(
            dissimilarity_matrices.0.clone(), 
            dissimilarity_matrices.1.clone(), 
            dissimilarity_matrices.0.size, 
            dissimilarity_matrices.1.size, 
            OrderedFloat(3.9), 
            OrderedFloat(0.0), 
            PrimeOrderFieldOperator::new(2), 
            subspace_order_operator.clone(),
            2 as usize
        );
        let oracle_factored = oracle.factor_from_arc(); 
        let oracle_wrapper = RelativeBoundaryMatrixOracleWrapper::new(oracle.clone(), oracle_factored); 
        let minor_keys = &oracle.get_sorted_keys_major_or_minor(false); 
        let order_operator = oracle_wrapper.source_comb_order_operator_ref(); 

        // CASE 1: relative cycle identification  
        // recall that a relative cycle is a chain in K with (possibly trivial) boundary in X0
        // with this data and threshold, we expext a basis for relative cycles to be a union of the following bases: 
        //      - a basis for all 0-chains in K ... a standard basis for this is {a, b, ..., j}
        //      - a basis for all 1-chains in X0 ... a standard basis for this is {[bi], [fj], [ij]}
        //      - a basis for all absolute cycles in K ... by inspection we see there are two! 
        // so we expexct a basis for all relative cycles to have rank 15
        let mut relative_cycles: Vec<SimplexFiltered<OrderedFloat<f64>>> = Vec::new(); 
        for key in minor_keys { 
            if order_operator.is_relative_cycle(&key) { 
                relative_cycles.push(key.clone()); 
            }
        }

        let num_rel_cycles = relative_cycles.iter().count(); 
        assert!(num_rel_cycles == 15); 

        // CASE 2: comparing relative cycles with different filtration
        let points = &relative_cycles[0..10]; 
        let chain = &relative_cycles[num_rel_cycles-1]; 
        for x in points { 
            assert!(order_operator.judge_lt(x, chain)); 
        }

        // Case 3: lexicographically comparing relative cycles with equivalent filtration
        let simp_bi = &relative_cycles[11]; 
        let chain_bi_fj_ij = &relative_cycles[12]; 
        let a = order_operator.relative_cycle_birth(simp_bi).unwrap(); 
        let b = order_operator.relative_cycle_birth(chain_bi_fj_ij).unwrap(); 
        assert_eq!(a, b); 
        assert!(order_operator.judge_lt(simp_bi, chain_bi_fj_ij)); 
    } 

    #[test]
    /// 
    /// Ensure that `TargetColumnIndexOrderOperator`: 
    /// 
    /// - correctly identifies relative boundaries 
    /// - correctly compares relative boundaries with different filtration 
    /// - lexicographically compares relative boundaries with equivalent filtration
    /// 
    fn test_relative_boundary_birth() { 
        let test_data = generate_circle_test_data(false, false); 
        let dissimilarity_matrices = generate_dissimilarity_matrices( 
            test_data.0, 
            test_data.1
        ); 
        // let max dissimilarity be 4.2
        // since we are using "generate_circle_test_data" with subset data {i, j}, then at this threshold the full space contains: 
        //      - the points {a, ..., j}
        //      - the edges {[ab], [bc], [cd], ..., [gh], [ha], [bi], [fj], [ij]}
        //      - the triangles {[abi], [bci], [efj], [fgj]}
        // and the subspace contians: 
        //      - the points {i, j}
        //      - the edge [ij]
        let subspace_order_operator = OrderOperatorSubComplexFiltrationSimplices::new(
            dissimilarity_matrices.1.clone(), 
            OrderedFloat(0.0), 
            OrderedFloat(4.2),
            None
        );
        let oracle = RelativeBoundaryMatrixOracle::new(
            dissimilarity_matrices.0.clone(), 
            dissimilarity_matrices.1.clone(), 
            dissimilarity_matrices.0.size, 
            dissimilarity_matrices.1.size, 
            OrderedFloat(4.2), 
            OrderedFloat(0.0), 
            PrimeOrderFieldOperator::new(2), 
            subspace_order_operator,
            2 as usize
        );
        let oracle_factored = oracle.factor_from_arc(); 
        let oracle_wrapper = RelativeBoundaryMatrixOracleWrapper::new(oracle.clone(), oracle_factored);  
        let minor_keys = &oracle.get_sorted_keys_major_or_minor(false); 
        let order_operator = oracle_wrapper.target_comb_order_operator_ref();

        // CASE 1: relative boundary identification  
        // recall that a relative boundary is a relative cycle which may be written as a = b + c where: 
        //      - b is a (p+1)-chain in K 
        //      - c is a p-chain in X0
        //      - in words, a relative boundary in quotient space K/X0 differs from an absolute boundary in K by a chain in X0
        // with this data and threshold, we expext a basis for relative boundaries to be a union of the following bases: 
        //      - a basis for all chains in X0 ... a standard basis is {i, j, [ij]}
        //      - a basis for all absolute boundaries in K
        //          a) standard basis for 1-boundaries is just the points {a, b, ..., j}
        //          b) there are four 2-boundaries which are not homologous
        // so we expexct a basis for all relative cycles to have rank 15
        let mut relative_boundaries: Vec<SimplexFiltered<OrderedFloat<f64>>> = Vec::new(); 
        for key in minor_keys { 
            if order_operator.is_relative_boundary(&key) { 
                relative_boundaries.push(key.clone()); 
            }
        }
        let num_rel_boundaries = relative_boundaries.iter().count(); 
        assert!(num_rel_boundaries == 15);  

        // CASE 2: comparing relative boundaries with different filtration
        // 1-boundaries are linear combinations of 0-simplices, and thus all have diameter/filtration = `dissimilarity_min`
        let one_boundaries = &relative_boundaries[0..10]; 
        let two_boundary = &relative_boundaries[num_rel_boundaries-1]; 
        for x in one_boundaries { 
            assert!(order_operator.judge_lt(x, two_boundary)); 
        }

        // Case 3: lexicographically comparing relative boundaries with equivalent filtration
        // all 2-simplices in K are born at the exact same time, assert their boundaries are ordered lexicographically
        let chain_ai_ab_bi = &relative_boundaries[num_rel_boundaries-4];
        let chain_ci_bc_bi = &relative_boundaries[num_rel_boundaries-3];
        let chain_ej_ef_fj = &relative_boundaries[num_rel_boundaries-2];
        let chain_gj_fg_fj = &relative_boundaries[num_rel_boundaries-1];
        let a = order_operator.relative_boundary_birth(chain_ai_ab_bi).unwrap();
        let b = order_operator.relative_boundary_birth(chain_ci_bc_bi).unwrap();  
        let c = order_operator.relative_boundary_birth(chain_ej_ef_fj).unwrap();
        let d = order_operator.relative_boundary_birth(chain_gj_fg_fj).unwrap();  
        assert!(a == b && b == c && c == d); 
        assert!(order_operator.judge_lt(chain_ai_ab_bi, chain_ci_bc_bi)); 
        assert!(order_operator.judge_lt(chain_ci_bc_bi, chain_ej_ef_fj)); 
        assert!(order_operator.judge_lt(chain_ej_ef_fj, chain_gj_fg_fj)); 

    }

    #[test] 
    ///
    /// Ensure that (filtered) relative cycle and relative boundary bases computed, respectively, by 
    /// `SourceColumnIndexOrderOperator` and `TargetColumnIndexOrderOperator` are sorted correctly. 
    /// 
    fn test_filtered_relative_cycles_and_boundaries() { 
        let test_data = generate_circle_test_data(true, false); 
        let dissimilarity_matrices = generate_dissimilarity_matrices( 
            test_data.0, 
            test_data.1
        ); 
        // let max dissimilarity be 3.9
        // since we are using "generate_circle_test_data" with subset data {b, f, i, j}, then at this threshold the full space contains: 
        //      - the points {a, ..., j}
        //      - the edges {[ab], [bc], [cd], ..., [gh], [ha], [bi], [fj], [ij]}
        // and the subspace contians: 
        //      - the points {b, f, i, j}
        //      - the edges {[bi], [fj], [ij]}
        let subspace_order_operator = OrderOperatorSubComplexFiltrationSimplices::new(
            dissimilarity_matrices.1.clone(), 
            OrderedFloat(0.0), 
            OrderedFloat(3.9),
            None
        );
        let oracle = RelativeBoundaryMatrixOracle::new(
            dissimilarity_matrices.0.clone(), 
            dissimilarity_matrices.1.clone(), 
            dissimilarity_matrices.0.size, 
            dissimilarity_matrices.1.size, 
            OrderedFloat(3.9), 
            OrderedFloat(0.0), 
            PrimeOrderFieldOperator::new(2), 
            subspace_order_operator,
            2 as usize
        );
        let oracle_factored = oracle.factor_from_arc(); 
        let oracle_wrapper = RelativeBoundaryMatrixOracleWrapper::new(oracle.clone(), oracle_factored);  
        let source_order_operator = oracle_wrapper.source_comb_order_operator_ref(); 
        let target_order_operator = oracle_wrapper.target_comb_order_operator_ref();

        // this is the exact same data as `test_relative_cycle_birth()`, assert the same result here! 
        let filtered_rel_cycle_basis = oracle_wrapper.filtered_relative_cycle_basis(); 
        assert!(filtered_rel_cycle_basis.clone().iter().count() == 15);

        // assert dimension of relative boundary basis leq to dimension of relative cycle basis 
        let filtered_rel_bound_basis = oracle_wrapper.filtered_relative_boundary_basis(); 
        assert!(filtered_rel_bound_basis.clone().iter().count() <= 15); 

        // assert each basis is actually filtered 
        let mut previous_diameter: Option<OrderedFloat<f64>> = None; 
        for rel_cycle in filtered_rel_cycle_basis { 
            let current_diameter = source_order_operator.relative_cycle_birth(&rel_cycle).unwrap(); 
            if previous_diameter.is_some() { 
                assert!(previous_diameter.unwrap() <= current_diameter)
            }
            previous_diameter = Some(current_diameter); 
        }
        previous_diameter = None; 
        for rel_bound in filtered_rel_bound_basis { 
            let current_diameter = target_order_operator.relative_boundary_birth(&rel_bound).unwrap(); 
            if previous_diameter.is_some() { 
                assert!(previous_diameter.unwrap() <= current_diameter)
            }
            previous_diameter = Some(current_diameter); 
        }
    }

    //  ===========================================================
    //  Test computing: 
    //  - representatives/bars for relative homology classes
    //  - birth filtration with `SourceColumnIndexOrderOperator`
    //  - death filtration with `TargetColumnIndexOrderOperator`
    //  =========================================================== 

    #[test] 
    ///
    /// Case: Dimension of bases for relative homolgy. 
    /// 
    fn test_prh_basis_dimension() { 
        let test_data = generate_circle_test_data(false, false); 
        let dissimilarity_matrices = generate_dissimilarity_matrices( 
            test_data.0, 
            test_data.1
        ); 
        // Let max dissimilarity be 2*sqrt(2). 
        // Since we are using "generate_circle_test_data" with subset data {i, j}, then at this threshold: 
        //      - the full space has points {a, ..., j} and edge [ij]
        //      - the subspace has points {i,j} and edge [ij]
        let two: f64 = 2.0; 
        let threshold = two * two.sqrt(); 
        let oracle = create_oracle_from_data(
            dissimilarity_matrices.0.clone(), 
            dissimilarity_matrices.1.clone(), 
            OrderedFloat(threshold), 
            None
        ); 
        let oracle_factored = oracle.factor_from_arc(); 
        let oracle_wrapper = RelativeBoundaryMatrixOracleWrapper::new(oracle.clone(), oracle_factored);  

        // compute filtered basis for: 
        // relative cycles 
        let filtered_relative_cycles = oracle_wrapper.filtered_relative_cycle_basis(); 
        let count_rel_cycles = filtered_relative_cycles.iter().count(); 
        // relative boundaries 
        let filtered_relative_boundaries = oracle_wrapper.filtered_relative_boundary_basis(); 
        let count_rel_boundaries = filtered_relative_boundaries.iter().count(); 
        // relative homological generators 
        let relative_homology_basis = oracle_wrapper.essential_cycles(); 
        let count_rel_generators = relative_homology_basis.iter().count(); 

        // check that dimension of these bases are correct 
        assert!(count_rel_boundaries == 3); 
        assert!(count_rel_generators == 8); 
        assert_eq!(count_rel_cycles, count_rel_boundaries + count_rel_generators);  

        // for each generator ... 
        let points_not_in_subspace = &oracle.get_sorted_keys_major_or_minor(false)[0..count_rel_generators];
        println!(); 
        println!(" =========================== "); 
        for i in 0..count_rel_generators { 
            let key_of_generator = relative_homology_basis[i].clone().0; 
            let current_generator = oracle_wrapper.umatch.comb_domain().view_minor_descend(key_of_generator.clone()).collect_vec();
            // print for reference 
            println!(" key of generator is . . . {:?}", vec![ (key_of_generator, Ratio::from(1 as isize)) ].chain_to_string(oracle_wrapper.umatch.ring_operator()) ); 
            println!(" the generator is . . . {:?}", current_generator.chain_to_string(oracle_wrapper.umatch.ring_operator()) );  
            println!(); 
            // assert that they are correct 
            // this is a simple example ... we expect `filtered_relative_homology_basis` to be a filtered basis for all zero-simplices (points) not in the subspace
            assert_eq!(current_generator.clone().iter().count(), 1); 
            // assert that the simplex is a point 
            assert_eq!(current_generator[0].0.num_vertices(), 1); 
            // assert that the simplex is the corresponding point of `points_not_in_subspace`
            let current_point = points_not_in_subspace[i].clone();
            assert_eq!(current_generator[0].0, current_point);
        }
    }

    #[test]
    ///
    /// Collapsing an absolute cycle:
    ///
    /// Case (1): an absolute cycle contained entirely in K_0. 
    /// Case (2): an absolute cycle contained entirely in K \ K_0. 
    /// 
    /// Require (1): the absolute cycle corresponds to a trivial bar/class in relative homology. 
    /// Require (2): the absolute cycle is a nontrivial relative homology class of K/K_0 with the same bar as the standard PH barcode of K.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   
    /// 
    fn test_absolute_cycle_in_subspace_or_full_space() { 
        let test_data = generate_side_by_side_circles_test_data(); 
        let dissimilarity_matrices = generate_dissimilarity_matrices( 
            test_data.0, 
            test_data.1
        ); 
        // Let max dissimilarity be 2*sqrt(2). 
        // Since we are using "generate_side_by_side_circles_test_data" then at this threshold: 
        //      - the full space has points {a, ..., h} and edges {[ab], [bc], [cd], [da], [ef], [fg], [gh], [he], [bh]}
        //      - the subspace has points {a, b, c, d} and edges {[ab], [bc], [cd], [da]}
        let two: f64 = 2.0; 
        let threshold = two * two.sqrt(); 
        let oracle = create_oracle_from_data(
            dissimilarity_matrices.0.clone(), 
            dissimilarity_matrices.1.clone(), 
            OrderedFloat(threshold), 
            None
        ); 
        let oracle_factored = oracle.factor_from_arc(); 
        let oracle_wrapper = RelativeBoundaryMatrixOracleWrapper::new(oracle.clone(), oracle_factored);

        // CASE 1: assert that the absolute cycle in the subcomplex is born and subsumed at `threshold`
        let key_for_generator_of_subsumed_class = oracle_wrapper.filtered_relative_cycle_basis()[11].clone(); 
        let birth = oracle_wrapper.source_comb_order_operator.relative_cycle_birth(&key_for_generator_of_subsumed_class.clone());  
        let death = oracle_wrapper.target_comb_order_operator.relative_boundary_birth(&key_for_generator_of_subsumed_class.clone());
        assert!(birth == death); 
        println!(); 
        println!(" =========================== "); 
        let current_generator = oracle_wrapper.umatch.comb_domain().view_minor_descend(key_for_generator_of_subsumed_class.clone()).collect_vec(); 
        println!(" key for the subsumed class is . . . {:?}", vec![ (key_for_generator_of_subsumed_class, Ratio::from(1 as isize)) ].chain_to_string(oracle_wrapper.umatch.ring_operator()) ); 
        println!(" the generator is . . . {:?}", current_generator.chain_to_string(oracle_wrapper.umatch.ring_operator()) );   

        // CASE 2: assert that the absolute cycle not in the subcomplex generates a relative homology class born at 'threshold'
        let relative_homology_basis = oracle_wrapper.essential_cycles(); 
        let count_rel_generators = relative_homology_basis.iter().count(); 
        assert_eq!(count_rel_generators, 1); 
        assert_eq!(oracle_wrapper.source_comb_order_operator.relative_cycle_birth(&relative_homology_basis[0].0).unwrap(), threshold); 
        // print the generator for reference 
        // this is a simple example ... 
        // - we expect `filtered_relative_homology_basis` to contain only ONE generator
        // - we expect the generator to be a chain of 1-simplices giving the circle on the right
        let gen_key = relative_homology_basis[0].clone().0;  
        let generator = oracle_wrapper.umatch.comb_domain().view_minor_descend(gen_key.clone()).collect_vec();
        println!();
        println!(" =========================== ");
        println!(" key for the nontrivial class is . . . {:?}", vec![ (gen_key, Ratio::from(1 as isize)) ].chain_to_string(oracle_wrapper.umatch.ring_operator()) );
        println!(" the generator is . . . {:?}", generator.chain_to_string(oracle_wrapper.umatch.ring_operator()) );  
        // assert that the chain gives a circle with for points 
        assert_eq!(generator.clone().iter().count(), 4);
        // assert that it is a linear combination of 1-simplices 
        for simplex in generator.clone() { 
            assert_eq!(simplex.0.num_vertices(), 2);
        }
    }

    #[test]
    ///
    /// Collapsing an absolute cycle: 
    /// 
    /// Case: an absolute cycle in K_0 with custom filtration. 
    /// 
    /// Require: the absolute cycle gives a nontrivial bar in the PRH barcode if and only if the customized 
    /// filtration on K_0 causes the cycle to enter the subcomplex strictly after it is born as a PH class 
    /// and strictly before it dies as a PH class.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               
    /// 
    fn test_absolute_cycle_in_subspace_with_custom_filtration() { 
        let test_data = generate_side_by_side_circles_test_data(); 
        let dissimilarity_matrices = generate_dissimilarity_matrices( 
            test_data.0, 
            test_data.1
        ); 
        // Let max dissimilarity be 4. 
        // At this threshold, we expect all 1-homology to vanish! 
        let threshold = 4.0; 
        let two: f64 = 2.0;
        let two_root_two = two * two.sqrt();
        let custom_subcomplex_data = Some(vec![
            Some(OrderedFloat(3.0)), 
            Some(OrderedFloat(3.1)), 
            Some(OrderedFloat(3.2)), 
            Some(OrderedFloat(3.3)), 
            None, 
            None, 
            None, 
            None
        ]); 
        let oracle = create_oracle_from_data(
            dissimilarity_matrices.0.clone(), 
            dissimilarity_matrices.1.clone(), 
            OrderedFloat(threshold), 
            custom_subcomplex_data
        ); 
        // oracle and order operators
        let oracle_factored = oracle.factor_from_arc(); 
        let oracle_wrapper = RelativeBoundaryMatrixOracleWrapper::new(oracle.clone(), oracle_factored);
        let source_order_operator = oracle_wrapper.source_comb_order_operator.to_owned();
        let target_order_operator = oracle_wrapper.target_comb_order_operator.to_owned();

        // relative cycle basis 
        let relative_cycle_basis = oracle_wrapper.filtered_relative_cycle_basis(); 

        // assert that the absolute cycle in the subcomplex generates a class existing over the interval <2*sqrt(2), 3>.
        let key_for_generator_of_custom_class = relative_cycle_basis[8].clone(); 
        let mut birth = source_order_operator.relative_cycle_birth(&key_for_generator_of_custom_class.clone());  
        let mut death = target_order_operator.relative_boundary_birth(&key_for_generator_of_custom_class.clone());
        assert_eq!(birth.unwrap().into_inner(), two_root_two); 
        assert_eq!(death.unwrap().into_inner(), 3.3); 
        println!(); 
        println!(" =========================== "); 
        let current_generator = oracle_wrapper.umatch.comb_domain().view_minor_descend(key_for_generator_of_custom_class.clone()).collect_vec(); 
        println!(" key for the customized class is . . . {:?}", vec![ (key_for_generator_of_custom_class, Ratio::from(1 as isize)) ].chain_to_string(oracle_wrapper.umatch.ring_operator()) ); 
        println!(" the generator is . . . {:?}", current_generator.chain_to_string(oracle_wrapper.umatch.ring_operator()) );   

        // assert that the absolute cycle not in the subcomplex generates a class existing over the interval <2*sqrt(2), 4>.  
        let key_for_generator_of_standard_class = relative_cycle_basis[9].clone();
        birth = source_order_operator.relative_cycle_birth(&key_for_generator_of_standard_class.clone());  
        death = target_order_operator.relative_boundary_birth(&key_for_generator_of_standard_class.clone());
        assert_eq!(birth.unwrap().into_inner(), two_root_two); 
        assert_eq!(death.unwrap().into_inner(), 4.0);  
        println!();
        println!(" =========================== ");
        let current_generator = oracle_wrapper.umatch.comb_domain().view_minor_descend(key_for_generator_of_standard_class.clone()).collect_vec(); 
        println!(" key for the nontrivial class is . . . {:?}", vec![ (key_for_generator_of_standard_class, Ratio::from(1 as isize)) ].chain_to_string(oracle_wrapper.umatch.ring_operator()) );
        println!(" the generator is . . . {:?}", current_generator.chain_to_string(oracle_wrapper.umatch.ring_operator()) );  
        
    }

    #[test]
    ///
    /// Collapsing a subcomplex:
    /// 
    /// Case: A single, contiguous subcomplex is removed at a given time.  
    /// 
    /// Require: does not create a new relative homology class. 
    /// 
    fn test_removing_a_single_subcomplex_of_an_absolute_cycle_does_not_create_new_class() { 
        let test_data = generate_circle_test_data(true, false); 
        let dissimilarity_matrices = generate_dissimilarity_matrices( 
            test_data.0, 
            test_data.1
        ); 
        // Let max dissimilarity be 3.9. 
        // Since we are using "generate_circle_test_data" with subset data {b, f, i, j}, then at this threshold: 
        //      - the full space has points {a, ..., j} and edges {[ab], [bc], [cd] ..., [ha], [bi], [fj], [ij]}
        //      - the subspace has points {b, f, i, j} and edges {[bi], [fj], [ij]}
        let oracle = create_oracle_from_data(
            dissimilarity_matrices.0.clone(), 
            dissimilarity_matrices.1.clone(), 
            OrderedFloat(3.9), 
            None
        ); 
        let oracle_factored = oracle.factor_from_arc(); 
        let oracle_wrapper = RelativeBoundaryMatrixOracleWrapper::new(oracle.clone(), oracle_factored);   
        
        // relative homological generators 
        // NOTE: this example has two absolute homology classes, and two relative homology classes (ie no new class created!!)
        let relative_homology_basis = oracle_wrapper.essential_cycles(); 
        let count_rel_generators = relative_homology_basis.iter().count(); 
        assert_eq!(count_rel_generators, 2);

        println!(" =========================== "); 
        println!(); 
        for i in 0..count_rel_generators { 
            let key_of_generator = relative_homology_basis[i].clone().0; 
            let current_generator = oracle_wrapper.umatch.comb_domain().view_minor_descend(key_of_generator.clone()).collect_vec();
            // assert that subspace simplices are included in the cycle reps 
            assert!(current_generator.clone().iter().count() == 7); 
            // print for reference
            println!(" key of generator is . . . {:?}", vec![ (key_of_generator, Ratio::from(1 as isize)) ].chain_to_string(oracle_wrapper.umatch.ring_operator()) );
            println!(" the generator is . . . {:?}", current_generator.chain_to_string(oracle_wrapper.umatch.ring_operator()) );  
            println!(); 
        }
    }

    #[test]
    ///
    /// Collapsing a subcomplex: 
    /// 
    /// Case: two disjoint subcomplexes are removed. 
    /// 
    /// (NOTE: this extends inductively to a subcomplex made of n disjoint complexes)
    /// 
    /// Require: results in a new relative homology class which is NOT also an absolute homology class. 
    /// 
    fn test_removing_two_disjoint_subcomplexes_of_an_absolute_cycle_creates_one_new_class() { 
        let test_data = generate_circle_test_data(false, true); 
        let dissimilarity_matrices = generate_dissimilarity_matrices( 
            test_data.0, 
            test_data.1
        ); 
        // Let max dissimilarity be 3.9. 
        // Since we are using "generate_circle_test_data" with subset data {b, f}, then at this threshold: 
        //      - the full space has points {a, ..., h} and edges {[ab], [bc], [cd] ..., [ha]}
        //      - the subspace has points {b, f}
        let oracle = create_oracle_from_data(
            dissimilarity_matrices.0.clone(), 
            dissimilarity_matrices.1.clone(), 
            OrderedFloat(3.9), 
            None
        ); 

        // oracle and order operators 
        let oracle_factored = oracle.factor_from_arc(); 
        let oracle_wrapper = RelativeBoundaryMatrixOracleWrapper::new(oracle.clone(), oracle_factored);  
        let keys = oracle_wrapper.mapping.get_key_list(); 
        
        // homological generators 
        // NOTE: this example has one absolute homology class, but two relative homology classes!
        let relative_homology_basis = oracle_wrapper.essential_cycles(); 
        let count_rel_generators = relative_homology_basis.iter().count();  
        assert_eq!(count_rel_generators, 2);
        let abs_generators = oracle_wrapper.umatch.matching_ref().filter_out_matched_minors(keys.into_iter());
        let num_subsumed_abs_classes = abs_generators.clone().into_iter()
            .filter_map(|x| oracle_wrapper.target_comb_order_operator.relative_boundary_birth(&x))
            .into_iter()
            .count(); 
        assert_eq!(abs_generators.count() - num_subsumed_abs_classes, 1); 

        // the relative homology class 
        let key_for_generator_of_relative_class = relative_homology_basis[0].clone(); 
        let generator_of_relative_class = oracle_wrapper.umatch.comb_domain().view_minor_descend(key_for_generator_of_relative_class.0.clone()).collect_vec();
        let mut birth = key_for_generator_of_relative_class.1;
        let mut length_of_generator = generator_of_relative_class.clone().iter().count(); 
        assert_eq!(birth.into_inner(), 3.8268343236508975); 
        assert_eq!(length_of_generator, 4);
        println!(); 
        println!(" =========================== ");
        println!(" key of generator for relative class is . . . {:?}", vec![ (key_for_generator_of_relative_class.0, Ratio::from(1 as isize)) ].chain_to_string(oracle_wrapper.umatch.ring_operator()) );
        println!(" the generator is . . . {:?}", generator_of_relative_class.chain_to_string(oracle_wrapper.umatch.ring_operator()) );  
        println!();

        // the absolute homology class 
        let key_for_generator_of_absolute_class = relative_homology_basis[1].clone(); 
        let generator_of_absolute_class = oracle_wrapper.umatch.comb_domain().view_minor_descend(key_for_generator_of_absolute_class.0.clone()).collect_vec();
        birth = key_for_generator_of_absolute_class.1;
        length_of_generator = generator_of_absolute_class.clone().iter().count();
        assert_eq!(birth.into_inner(), 3.8268343236508975); 
        assert_eq!(length_of_generator, 8); 
        println!(); 
        println!(" =========================== ");
        println!(" key of generator for absolute class is . . . {:?}", vec![ (key_for_generator_of_absolute_class.0, Ratio::from(1 as isize)) ].chain_to_string(oracle_wrapper.umatch.ring_operator()) );
        println!(" the generator is . . . {:?}", generator_of_absolute_class.chain_to_string(oracle_wrapper.umatch.ring_operator()) );  
        println!(); 
    } 
    
    #[test]
    ///
    /// Collapsing a subcomplex: 
    /// 
    /// Case: two disjoint subcomplexes with distinct custom filtration values are removed, where the max 
    /// of the custom filtration values is greater than the filtration parameter at which a chain bounded by 
    /// the removed complexes is born.
    /// 
    /// (NOTE: this extends inductively to a subcomplex made of n disjoint complexes)
    /// 
    /// Require: results in a new relative homology class (which is NOT also an absolute class) where 
    /// this new class is born at the larger of the two custom filtration values. 
    /// 
    fn test_removing_two_disjoint_subcomplexes_of_an_absolute_cycle_creates_one_new_class_custom_after() { 
        let test_data = generate_circle_test_data(false, true); 
        let dissimilarity_matrices = generate_dissimilarity_matrices( 
            test_data.0, 
            test_data.1
        ); 

        // Let max dissimilarity be 3.9. 
        // Since we are using "generate_circle_test_data" with subset data {b, f}, then at this threshold: 
        //      - the full space has points {a, ..., h} and edges {[ab], [bc], [cd] ..., [ha]}
        //      - the subspace has points {b, f}
        let custom_diameter_1 = OrderedFloat(3.85);
        let custom_diameter_2 = OrderedFloat(3.86); 
        let custom_subcomplex_data = Some(vec![
            None, 
            Some(custom_diameter_1), 
            None, 
            None, 
            None, 
            Some(custom_diameter_2), 
            None, 
            None
        ]);
        let a = OrderedFloat(3.8268343236508975); 
        let b = std::cmp::max(custom_diameter_1, custom_diameter_2); 
        let oracle = create_oracle_from_data(
            dissimilarity_matrices.0.clone(), 
            dissimilarity_matrices.1.clone(), 
            OrderedFloat(3.9), 
            custom_subcomplex_data
        ); 

        // oracle and order operators 
        let oracle_factored = oracle.factor_from_arc(); 
        let oracle_wrapper = RelativeBoundaryMatrixOracleWrapper::new(oracle.clone(), oracle_factored);  
        let keys = oracle_wrapper.mapping.get_key_list(); 
        
        // homological generators 
        // NOTE: this example has one absolute homology class, but two relative homology classes!
        let relative_homology_basis = oracle_wrapper.essential_cycles(); 
        let count_rel_generators = relative_homology_basis.iter().count();  
        assert_eq!(count_rel_generators, 2);
        let abs_generators = oracle_wrapper.umatch.matching_ref().filter_out_matched_minors(keys.into_iter());
        let num_subsumed_abs_classes = abs_generators.clone().into_iter()
            .filter_map(|x| oracle_wrapper.target_comb_order_operator.relative_boundary_birth(&x))
            .into_iter()
            .count(); 
        assert_eq!(abs_generators.count() - num_subsumed_abs_classes, 1); 

        // the relative homology class 
        let key_for_generator_of_relative_class = relative_homology_basis[1].clone(); 
        let generator_of_relative_class = oracle_wrapper.umatch.comb_domain().view_minor_descend(key_for_generator_of_relative_class.0.clone()).collect_vec();
        let mut birth = key_for_generator_of_relative_class.1;
        let mut length_of_generator = generator_of_relative_class.clone().iter().count(); 
        assert_eq!(birth.into_inner(), b.into_inner()); 
        assert_eq!(length_of_generator, 4);
        println!(); 
        println!(" =========================== ");
        println!(" key of generator for relative class is . . . {:?}", vec![ (key_for_generator_of_relative_class.0, Ratio::from(1 as isize)) ].chain_to_string(oracle_wrapper.umatch.ring_operator()) );
        println!(" the generator is . . . {:?}", generator_of_relative_class.chain_to_string(oracle_wrapper.umatch.ring_operator()) );  
        println!();

        // the absolute homology class 
        let key_for_generator_of_absolute_class = relative_homology_basis[0].clone(); 
        let generator_of_absolute_class = oracle_wrapper.umatch.comb_domain().view_minor_descend(key_for_generator_of_absolute_class.0.clone()).collect_vec();
        birth = key_for_generator_of_absolute_class.1;
        length_of_generator = generator_of_absolute_class.clone().iter().count();
        assert_eq!(birth.into_inner(), a.into_inner()); 
        assert_eq!(length_of_generator, 8); 
        println!(); 
        println!(" =========================== ");
        println!(" key of generator for absolute class is . . . {:?}", vec![ (key_for_generator_of_absolute_class.0, Ratio::from(1 as isize)) ].chain_to_string(oracle_wrapper.umatch.ring_operator()) );
        println!(" the generator is . . . {:?}", generator_of_absolute_class.chain_to_string(oracle_wrapper.umatch.ring_operator()) );  
        println!(); 
    }

    #[test]
    ///
    /// Collapsing a subcomplex: 
    /// 
    /// Case: two disjoint subcomplexes with distinct custom filtration values are removed, where the 
    /// max of the custom filtration values is less than the filtration parameter at which a chain bounded by 
    /// the removed complexes is born
    /// 
    /// (NOTE: this extends inductively to a subcomplex made of n disjoint complexes)
    /// 
    /// Require: results in a new relative homology class (which is NOT also an absolute class) where 
    /// this new class is born when the first chain bounded by the removed subcomplexes is born. 
    /// 
    fn test_removing_two_disjoint_subcomplexes_of_an_absolute_cycle_creates_one_new_class_custom_before() { 
        let test_data = generate_circle_test_data(false, true); 
        let dissimilarity_matrices = generate_dissimilarity_matrices( 
            test_data.0, 
            test_data.1
        ); 

        // Let max dissimilarity be 3.9. 
        // Since we are using "generate_circle_test_data" with subset data {b, f}, then at this threshold: 
        //      - the full space has points {a, ..., h} and edges {[ab], [bc], [cd] ..., [ha]}
        //      - the subspace has points {b, f}
        let custom_diameter_1 = OrderedFloat(1.0);
        let custom_diameter_2 = OrderedFloat(2.0); 
        let custom_subcomplex_data = Some(vec![
            None, 
            Some(custom_diameter_1), 
            None, 
            None, 
            None, 
            Some(custom_diameter_2), 
            None, 
            None
        ]);
        let a = OrderedFloat(3.8268343236508975);  
        let oracle = create_oracle_from_data(
            dissimilarity_matrices.0.clone(), 
            dissimilarity_matrices.1.clone(), 
            OrderedFloat(3.9), 
            custom_subcomplex_data
        ); 

        // oracle and order operators 
        let oracle_factored = oracle.factor_from_arc(); 
        let oracle_wrapper = RelativeBoundaryMatrixOracleWrapper::new(oracle.clone(), oracle_factored);  
        let keys = oracle_wrapper.mapping.get_key_list(); 
        
        // homological generators 
        // NOTE: this example has one absolute homology class, but two relative homology classes!
        let relative_homology_basis = oracle_wrapper.essential_cycles(); 
        let count_rel_generators = relative_homology_basis.iter().count();  
        assert_eq!(count_rel_generators, 2);
        let abs_generators = oracle_wrapper.umatch.matching_ref().filter_out_matched_minors(keys.into_iter());
        let num_subsumed_abs_classes = abs_generators.clone().into_iter()
            .filter_map(|x| oracle_wrapper.target_comb_order_operator.relative_boundary_birth(&x))
            .into_iter()
            .count(); 
        assert_eq!(abs_generators.count() - num_subsumed_abs_classes, 1); 

        // the relative homology class 
        let key_for_generator_of_relative_class = relative_homology_basis[0].clone(); 
        let generator_of_relative_class = oracle_wrapper.umatch.comb_domain().view_minor_descend(key_for_generator_of_relative_class.0.clone()).collect_vec();
        let mut birth = key_for_generator_of_relative_class.1;
        let mut length_of_generator = generator_of_relative_class.clone().iter().count(); 
        assert_eq!(birth.into_inner(), a.into_inner()); 
        assert_eq!(length_of_generator, 4);
        println!(); 
        println!(" =========================== ");
        println!(" key of generator for relative class is . . . {:?}", vec![ (key_for_generator_of_relative_class.0, Ratio::from(1 as isize)) ].chain_to_string(oracle_wrapper.umatch.ring_operator()) );
        println!(" the generator is . . . {:?}", generator_of_relative_class.chain_to_string(oracle_wrapper.umatch.ring_operator()) );  
        println!();

        // the absolute homology class 
        let key_for_generator_of_absolute_class = relative_homology_basis[1].clone(); 
        let generator_of_absolute_class = oracle_wrapper.umatch.comb_domain().view_minor_descend(key_for_generator_of_absolute_class.0.clone()).collect_vec();
        birth = key_for_generator_of_absolute_class.1;
        length_of_generator = generator_of_absolute_class.clone().iter().count();
        assert_eq!(birth.into_inner(), a.into_inner()); 
        assert_eq!(length_of_generator, 8); 
        println!(); 
        println!(" =========================== ");
        println!(" key of generator for absolute class is . . . {:?}", vec![ (key_for_generator_of_absolute_class.0, Ratio::from(1 as isize)) ].chain_to_string(oracle_wrapper.umatch.ring_operator()) );
        println!(" the generator is . . . {:?}", generator_of_absolute_class.chain_to_string(oracle_wrapper.umatch.ring_operator()) );  
    }


    #[test]
    ///
    /// Collapsing a simplex or chain (not an absolute cycle): 
    /// 
    /// Case: a subcomplex simplex
    /// 
    /// Require: the subcomplex simplex gives a trivial bar in the PRH barcode, where the 
    /// relative homology class emerges and dies at the diameter of the simplex.  
    /// 
    fn test_subcomplex_simplices_are_trivial_classes() { 
        let test_data = generate_circle_test_data(false, false); 
        let dissimilarity_matrices = generate_dissimilarity_matrices( 
            test_data.0, 
            test_data.1
        ); 

        // Let max dissimilarity be 2*sqrt(2). 
        // Since we are using "generate_circle_test_data" with subset data {i, j}, then at this threshold: 
        //      - the full space has points {a, ..., j} and edge [ij]
        //      - the subspace has points {i,j} and edge [ij]
        let two: f64 = 2.0; 
        let threshold = two * two.sqrt(); 
        let oracle = create_oracle_from_data(
            dissimilarity_matrices.0.clone(), 
            dissimilarity_matrices.1.clone(), 
            OrderedFloat(threshold), 
            None
        );

        // boundary oracle and wrapper 
        let oracle_factored = oracle.factor_from_arc(); 
        let oracle_wrapper = RelativeBoundaryMatrixOracleWrapper::new(oracle.clone(), oracle_factored);

        // the relative homology class for simplex [i,j] or [8, 9]
        let key = oracle_wrapper.filtered_relative_cycle_basis().last().unwrap().clone(); 
        let birth = oracle_wrapper.source_comb_order_operator.relative_cycle_birth(&key);
        let death = oracle_wrapper.target_comb_order_operator.relative_boundary_birth(&key); 
        assert_eq!(birth.unwrap().into_inner(), threshold); 
        assert_eq!(birth, death); 
        println!(); 
        println!(" =========================== ");
        println!(" key of generator for relative class is . . . {:?}", vec![ (key.clone(), Ratio::from(1 as isize)) ].chain_to_string(oracle_wrapper.umatch.ring_operator()) );
        let generator = oracle_wrapper.umatch.comb_domain().view_minor_descend(key).collect_vec(); 
        println!(" the generator is . . . {:?}", generator.chain_to_string(oracle_wrapper.umatch.ring_operator()) );  
        println!();
    }

    #[test]
    ///
    /// Collapsing a simplex or chain (not an absolute cycle): 
    /// 
    /// Case: a subcomplex simplex with custom filtration
    /// 
    /// Require: the subcomplex simplex gives a trivial bar in the PRH barcode, where the 
    /// relative homology class emerges and dies at the CUSTOM diameter of the simplex, or 
    /// the scale diameter of the simplex if custom diameter <= scale diameter.  
    /// 
    fn test_subcomplex_simplices_with_custom_filtration_are_trivial_classes() { 
        let test_data = generate_circle_test_data(false, false); 
        let dissimilarity_matrices = generate_dissimilarity_matrices( 
            test_data.0, 
            test_data.1
        ); 

        // Let max dissimilarity be 3.0
        // Since we are using "generate_circle_test_data" with subset data {i, j}, then at this threshold: 
        //      - the full space has points {a, ..., j} and edge [ij]
        //      - the subspace has points {i,j} and edge [ij]
        let two: f64 = 2.0; 
        let two_root_two = two * two.sqrt(); 
        let custom_diameter_1 = OrderedFloat(2.85);
        let custom_diameter_2 = OrderedFloat(2.95); 
        let custom_subcomplex_data = Some(vec![
            None, 
            None, 
            None, 
            None, 
            None, 
            None, 
            None, 
            None, 
            Some(custom_diameter_1), 
            Some(custom_diameter_2)
        ]);
        let oracle = create_oracle_from_data(
            dissimilarity_matrices.0.clone(), 
            dissimilarity_matrices.1.clone(), 
            OrderedFloat(3.0), 
            custom_subcomplex_data
        );

        // NOTE: this check is correct, but not necessary as the code is currently written! 
        // determine the birth of the class by hand
        let a = std::cmp::max(custom_diameter_1, custom_diameter_2).into_inner(); 
        let b = match a > two_root_two { 
            true => { a }, 
            false => { two_root_two }
        }; 

        // boundary oracle and wrapper 
        let oracle_factored = oracle.factor_from_arc(); 
        let oracle_wrapper = RelativeBoundaryMatrixOracleWrapper::new(oracle.clone(), oracle_factored);

        // the relative homology class for simplex [i,j] or [8, 9]
        let key = oracle_wrapper.filtered_relative_cycle_basis().last().unwrap().clone(); 
        let birth = oracle_wrapper.source_comb_order_operator.relative_cycle_birth(&key);
        let death = oracle_wrapper.target_comb_order_operator.relative_boundary_birth(&key); 
        assert_eq!(birth.unwrap().into_inner(), b); 
        assert_eq!(birth, death); 
        println!(); 
        println!(" =========================== ");
        println!(" key of generator for relative class is . . . {:?}", vec![ (key.clone(), Ratio::from(1 as isize)) ].chain_to_string(oracle_wrapper.umatch.ring_operator()) );
        let generator = oracle_wrapper.umatch.comb_domain().view_minor_descend(key).collect_vec(); 
        println!(" the generator is . . . {:?}", generator.chain_to_string(oracle_wrapper.umatch.ring_operator()) );  
        println!();
    }

    #[test]
    ///
    /// Collapsing a simplex or chain (not an absolute cycle): 
    /// 
    /// Case: collapsing the boundary of a chain 
    /// 
    /// Require: the chain is a relative cycle with the class being born when the chain is born.   
    /// 
    fn test_collapsing_the_boundary_of_a_chain_creates_a_relative_homology_class() {
        let test_data = generate_line_test_data(); 
        let dissimilarity_matrices = generate_dissimilarity_matrices( 
            test_data.0, 
            test_data.1
        ); 
        // Let max dissimilarity be 2.0. 
        // Since we are using "generate_line_test_data", then at this threshold: 
        //      - the full space has points {a, b, c, d} and edges {[ab], [bc], [cd]}
        //      - the subspace has points {a, d} 
        let oracle = create_oracle_from_data(
            dissimilarity_matrices.0.clone(), 
            dissimilarity_matrices.1.clone(), 
            OrderedFloat(2.0), 
            None
        ); 
        let oracle_factored = oracle.factor_from_arc(); 
        let oracle_wrapper = RelativeBoundaryMatrixOracleWrapper::new(oracle.clone(), oracle_factored);  
        
        // relative homological generators 
        let relative_homology_basis = oracle_wrapper.essential_cycles(); 
        let count_rel_generators = relative_homology_basis.iter().count(); 

        // assert correct dimension of basis
        assert_eq!(count_rel_generators, 1);

        // assert correct birth diameter 
        let birth = oracle_wrapper.source_comb_order_operator.relative_cycle_birth(&relative_homology_basis[0].0); 
        assert_eq!(birth.unwrap().into_inner(), 2.0); 

        // print the generator for reference 
        // this is a simple example ... 
        // - we expect `filtered_relative_homology_basis` to contain only ONE generator
        // - we expect the generator to be a chain of 1-simplices with subspace endpoints identified together 
        let gen_key = relative_homology_basis[0].clone().0;  
        let generator = oracle_wrapper.umatch.comb_domain().view_minor_descend(gen_key.clone()).collect_vec();
        println!(" =========================== "); 
        println!(); 
        println!(" key of generator is . . . {:?}", vec![ (gen_key, Ratio::from(1 as isize)) ].chain_to_string(oracle_wrapper.umatch.ring_operator()) );
        println!(" the generator is . . . {:?}", generator.chain_to_string(oracle_wrapper.umatch.ring_operator()) );  
        
        // assert that the chain gives a line with three points (not in quotient topology)
        assert_eq!(generator.clone().iter().count(), 3);
        // assert that it is a linear combination of 1-simplices 
        for simplex in generator.clone() { 
            assert_eq!(simplex.0.num_vertices(), 2);
        }
    }

    #[test]
    ///
    /// Collapsing a simplex or chain (not an absolute cycle): 
    /// 
    /// Case: collapsing the boundary of a chain with customized subcomplex filtration.
    /// 
    /// Require: the chain is a relative cycle with the class being born when its boundary enters the subcomplex filtration.   
    /// 
    fn test_collapsing_the_boundary_of_a_chain_creates_a_relative_homology_class_custom() {
        let test_data = generate_line_test_data(); 
        let dissimilarity_matrices = generate_dissimilarity_matrices( 
            test_data.0, 
            test_data.1
        ); 
        // Let max dissimilarity be 3.0
        // Since we are using "generate_line_test_data", then at this threshold: 
        //      - the full space has points {a, b, c, d} and edges {[ab], [bc], [cd]}
        //      - the subspace has points {a, d} 
        let custom_diameter_1 = OrderedFloat(2.5); 
        let custom_diameter_2 = OrderedFloat(2.71828); 
        let custom_subcomplex_data = Some(vec![
            Some(custom_diameter_1),
            None, 
            None, 
            Some(custom_diameter_2)
        ]);
        let oracle = create_oracle_from_data(
            dissimilarity_matrices.0.clone(), 
            dissimilarity_matrices.1.clone(), 
            OrderedFloat(3.0), 
            custom_subcomplex_data
        ); 
        let oracle_factored = oracle.factor_from_arc(); 
        let oracle_wrapper = RelativeBoundaryMatrixOracleWrapper::new(oracle.clone(), oracle_factored);  
        
        // relative homological generators 
        let relative_homology_basis = oracle_wrapper.essential_cycles(); 
        let count_rel_generators = relative_homology_basis.iter().count(); 

        // assert correct dimension of basis
        assert_eq!(count_rel_generators, 1);

        // assert correct birth diameter 
        let birth = oracle_wrapper.source_comb_order_operator.relative_cycle_birth(&relative_homology_basis[0].0); 
        let custom_diameter_max = std::cmp::max(custom_diameter_1, custom_diameter_2);
        assert_eq!(birth.unwrap(), std::cmp::max(OrderedFloat(2.0), custom_diameter_max)); 

        // print the generator for reference 
        // this is a simple example ... 
        // - we expect `filtered_relative_homology_basis` to contain only ONE generator
        // - we expect the generator to be a chain of 1-simplices with subspace endpoints identified together 
        let gen_key = relative_homology_basis[0].clone().0;  
        let generator = oracle_wrapper.umatch.comb_domain().view_minor_descend(gen_key.clone()).collect_vec();
        println!(" =========================== "); 
        println!(); 
        println!(" key of generator is . . . {:?}", vec![ (gen_key, Ratio::from(1 as isize)) ].chain_to_string(oracle_wrapper.umatch.ring_operator()) );
        println!(" the generator is . . . {:?}", generator.chain_to_string(oracle_wrapper.umatch.ring_operator()) );  
        
        // assert that the chain gives a line with three points (not in quotient topology)
        assert_eq!(generator.clone().iter().count(), 3);
        // assert that it is a linear combination of 1-simplices 
        for simplex in generator.clone() { 
            assert_eq!(simplex.0.num_vertices(), 2);
        }
    }

    //  ===========================================================
    //  Tests with randomized data
    //  =========================================================== 

    #[test]
    /// 
    /// We randomly generate data and construct D, the boundary matrix of a filtered, quotient chain complex. 
    /// 
    /// Case: we use the U-match TM = DS to determine the dimension of a basis for relative cycles. Specifically, 
    /// we perform the matrix-vector multiplication D * S[:,c] to determine the boundary of the chain given by the 
    /// column vector S[:,c]. We check by cases, and increment the count of relative cycles. 
    /// 
    /// Require: the function `RelativeBoundaryMatrixOracleWrapper::filtered_relative_cycle_basis` returns a basis 
    /// with identical dimension.
    /// 
    fn test_computing_relative_cycle_basis_randomized_data() { 
        // randomly generate data 
        let num_points = 50 as usize; 
        let num_subset_points = rand::thread_rng().gen_range((2 as usize)..(num_points)-1);
        let (points, points_subset) = generate_random_test_data_on_disk_at_origin(
            5.0, 
            num_points, 
            num_subset_points
        ); 

        // create dissimilarity matrices and boundary oracle
        let (dissimilarity_full_space, dissimilarity_subspace) = generate_dissimilarity_matrices(points, points_subset);
        let oracle = create_oracle_from_data(
            dissimilarity_full_space, 
            dissimilarity_subspace, 
            OrderedFloat(2.5), 
            None
        ); 
        let keys = oracle.get_key_list();
        let subcomplex_order_operator = oracle.order_operator_sub_complex_ref().to_owned();
        let num_simplices = keys.clone().iter().count(); 
        println!(" =============== "); 
        println!("- The full complex has {:?} simplices.", num_simplices); 
        let num_simplices_subcomplex = keys.clone().iter().filter_map(|x| subcomplex_order_operator.diameter(&x.vertices)).count();
        println!("- The subcomplex has {:?} simplices.", num_simplices_subcomplex); 
        
        // factor oracle and create the oracle wrapper (this gives us access to methods to compute relative homology!)
        let start = std::time::Instant::now(); 
        let oracle_factored = oracle.factor_from_arc(); 
        let time_to_factor = start.elapsed().as_secs_f64();
        println!("- It took {:?} seconds to compute the first U-match", time_to_factor); 
        let oracle_wrapper = RelativeBoundaryMatrixOracleWrapper::new(oracle.clone(), oracle_factored);
        
        // get matrices we need from the first U-match
        let source = oracle_wrapper.umatch.comb_domain(); 
        let matching = oracle_wrapper.umatch.matching_ref(); 
        let boundary = oracle_wrapper.umatch.mapping_ref(); 
        
        // get some information about homology
        let keys_to_generate_filtered_absolute_cycle_basis = matching.filter_out_matched_minors(keys.clone()).collect_vec(); 
        let keys_to_generate_filtered_relative_cycle_basis = oracle_wrapper.filtered_relative_cycle_basis(); 
        let num_abs_cycles = keys_to_generate_filtered_absolute_cycle_basis.len(); 
        let num_rel_cycles = keys_to_generate_filtered_relative_cycle_basis.len(); 
        
        // now we iterate over the keys of the source COMB (NOT the keys which generate the relative cycle basis)
        let mut count_abs_cycles: usize = 0; 
        let mut count_rel_cycles: usize = 0; 
        for key in keys { 
            // the chain S[:, key] 
            let chain = source.view_minor_descend(key.clone()).collect_vec();
            // the boundary of the chain S[:, key]
            let boundary_vec = vector_matrix_multiply_minor_descend_simplified(
                chain.clone(), 
                boundary, 
                boundary.ring_operator, 
                boundary.row_index_order_operator.clone()
            ).collect_vec(); 
            let length_of_boundary_vec = boundary_vec.len(); 
            let is_boundary_vec_in_subcomplex = boundary_vec.clone().iter().filter_map(|x| subcomplex_order_operator.diameter(&x.0.vertices)).count() == length_of_boundary_vec;
            // if the boundary of the chain maps into the subcomplex ... 
            if is_boundary_vec_in_subcomplex { 
                count_rel_cycles = count_rel_cycles + 1;  
                // absolute cycle case
                if length_of_boundary_vec == 0 { 
                    count_abs_cycles = count_abs_cycles + 1;
                    assert!(keys_to_generate_filtered_relative_cycle_basis.contains(&key));
                    assert!(keys_to_generate_filtered_absolute_cycle_basis.contains(&key)); 
                // (ONLY) relative cycle case 
                } else { 
                    // NOTE: this asserts that any chain with a nontrivial, subcomplex boundary is also a relative (but not absolute) cycle!
                    assert!(keys_to_generate_filtered_relative_cycle_basis.contains(&key));
                    assert!(!keys_to_generate_filtered_absolute_cycle_basis.contains(&key)); 
                }
            }
        }
        // A FEW FINAL CHECKS:
        // due to the assertions above, these should ALWAYS pass if we make it here! 
        assert_eq!(count_abs_cycles, num_abs_cycles);
        assert_eq!(count_rel_cycles, num_rel_cycles);
        println!("- Our algorithm found {:?} relative cycles.", num_rel_cycles); 
        println!("- Our brute force test, which repeatedly performes matrix-vector multiplication and analyzes the result, found {:?} relative cycles.", count_rel_cycles); 
        print!(" =============== ");
    }

    #[test]
    ///
    /// We randomly generate data and construct D, the boundary matrix of a filtered, quotient chain complex. 
    /// 
    /// Case: we use the U-match TM = DS to determine the dimension of a basis for relative boundaries. Specifically, 
    /// we perform the matrix-vector multiplication D * T[:,r] to determine the boundary of the chain given by the 
    /// column vector T[:,r]. We check by cases, and increment the count of relative boundaries. 
    /// 
    /// Require: the function `RelativeBoundaryMatrixOracleWrapper::filtered_relative_boundary_basis` returns a basis 
    /// with identical dimension.
    /// 
    fn test_computing_relative_boundary_basis_randomized_data() { 
        // randomly generate data 
        let num_points = 50 as usize; 
        let num_subset_points = rand::thread_rng().gen_range((2 as usize)..(num_points)-1);
        let (points, points_subset) = generate_random_test_data_on_disk_at_origin(
            5.0, 
            num_points, 
            num_subset_points
        ); 

        // create dissimilarity matrices and boundary oracle
        let (dissimilarity_full_space, dissimilarity_subspace) = generate_dissimilarity_matrices(points, points_subset);
        let oracle = create_oracle_from_data(
            dissimilarity_full_space, 
            dissimilarity_subspace, 
            OrderedFloat(2.5), 
            None
        ); 
        let keys = oracle.get_key_list();
        let subcomplex_order_operator = oracle.order_operator_sub_complex_ref().to_owned();
        let num_simplices = keys.clone().iter().count(); 
        println!(" =============== "); 
        println!("- The full complex has {:?} simplices.", num_simplices); 
        let num_simplices_subcomplex = keys.clone().iter().filter_map(|x| subcomplex_order_operator.diameter(&x.vertices)).count();
        println!("- The subcomplex has {:?} simplices.", num_simplices_subcomplex); 
        
        // factor oracle and create the oracle wrapper (this gives us access to methods to compute relative homology!)
        let start = std::time::Instant::now(); 
        let oracle_factored = oracle.factor_from_arc(); 
        let time_to_factor = start.elapsed().as_secs_f64();
        println!("- It took {:?} seconds to compute the first U-match", time_to_factor); 
        let oracle_wrapper = RelativeBoundaryMatrixOracleWrapper::new(oracle.clone(), oracle_factored);
        
        // get matrices we need from the first U-match
        let target = oracle_wrapper.umatch.comb_codomain(); 
        let matching = oracle_wrapper.umatch.matching_ref(); 
        let boundary = oracle_wrapper.umatch.mapping_ref(); 
        
        // get some information about homology
        let keys_to_generate_filtered_absolute_boundary_basis = matching.filter_only_matched_majors(keys.clone()).collect_vec(); 
        let keys_to_generate_filtered_absolute_cycle_basis = matching.filter_out_matched_minors(keys.clone()).collect_vec(); 
        let keys_to_generate_filtered_relative_boundary_basis = oracle_wrapper.filtered_relative_boundary_basis(); 
        let num_rel_boundaries = keys_to_generate_filtered_relative_boundary_basis.len(); 
        
        // now we iterate over the keys of the target COMB (NOT the keys which generate the relative boundary basis)
        let mut count_rel_boundaries: usize = 0; 
        for key in keys { 
            // the chain T[:, key] 
            let chain = target.view_minor_descend(key.clone()).collect_vec();
            let length_of_chain = chain.len(); 
            let is_chain_in_subcomplex = chain.clone().iter().filter_map(|x| subcomplex_order_operator.diameter(&x.0.vertices)).count() == length_of_chain;
            // the boundary of the chain T[:, key]
            let boundary_vec = vector_matrix_multiply_minor_descend_simplified(
                chain.clone(), 
                boundary, 
                boundary.ring_operator, 
                boundary.row_index_order_operator.clone()
            ).collect_vec(); 
            let length_of_boundary_vec = boundary_vec.len(); 
            let is_boundary_vec_in_subcomplex = boundary_vec.clone().iter().filter_map(|x| subcomplex_order_operator.diameter(&x.0.vertices)).count() == length_of_boundary_vec;
            // if the boundary of the chain maps into the subcomplex ... 
            if is_boundary_vec_in_subcomplex { 
                // CASE 1: the boundary is trivial
                if length_of_boundary_vec == 0 { 
                    // CASE 1.1: ALL absolute boundaries 
                    if keys_to_generate_filtered_absolute_boundary_basis.contains(&key) {
                        count_rel_boundaries = count_rel_boundaries + 1;
                        assert!(keys_to_generate_filtered_relative_boundary_basis.contains(&key));
                    // CASE 1.2 essential absolute cycle contained ENTIRELY in the subcomplex
                    } else {
                        if is_chain_in_subcomplex { 
                            count_rel_boundaries = count_rel_boundaries + 1;
                            assert!(keys_to_generate_filtered_absolute_cycle_basis.contains(&key));
                            assert!(keys_to_generate_filtered_relative_boundary_basis.contains(&key));
                        }
                    }
                }
                // CASE 2: the chain is entirely contained in the subcomplex and its boundary is nontrivial
                else { 
                    if is_chain_in_subcomplex { // this check is crucial ... chains which are not members of the subcomplex but have a boundary in the subcomplex are NOT relative boundaries!
                        count_rel_boundaries = count_rel_boundaries + 1;
                        assert!(keys_to_generate_filtered_relative_boundary_basis.contains(&key));
                        assert!(!keys_to_generate_filtered_absolute_boundary_basis.contains(&key));
                    }
                }
            }
        }
        // A FEW FINAL CHECKS:
        // due to the assertions above, these should ALWAYS pass if we make it here! 
        assert_eq!(count_rel_boundaries, num_rel_boundaries);
        println!("- Our algorithm found {:?} relative boundaries.", num_rel_boundaries); 
        println!("- Our brute force test, which repeatedly performes matrix-vector multiplication and analyzes the result, found {:?} relative boundaries.", count_rel_boundaries); 
        print!(" =============== ");
    }

    // TODO: These tests should be moved to clique_filtered.rs in oat_python/src once we are done with the refactor!!

    // NOTE for developers : the code required for this unit test is still in development ... see oat_rust/src/algebra/chains/relative/oracle.rs
    // #[test]
    // ///  
    // /// We randomly generate data and construct D, the boundary matrix of a filtered, quotient chain complex. 
    // /// 
    // /// Case: extracting a persistence moudle using the keys returned from `RelativeBoundaryMatrixOracleWrapper::filtered_relative_cycle_basis`
    // /// 
    // /// Require: the matched basis contains all boundry representatives, and spans all relative cycles and relative boundaries. 
    // /// 
    // /// NOTE: we compute the matched basis via a lazy, and unsorted COMB product T^{-1}S
    // /// 
    // fn test_computing_lazy_matched_basis_matrix_randomized_data() { 
    //     // randomly generate data 
    //     let num_points = 50 as usize; 
    //     let num_subset_points = rand::thread_rng().gen_range((2 as usize)..(num_points)-1);
    //     let (points, points_subset) = generate_random_test_data_on_disk_at_origin(
    //         5.0, 
    //         num_points, 
    //         num_subset_points
    //     ); 

    //     // create dissimilarity matrices and boundary oracle
    //     let (dissimilarity_full_space, dissimilarity_subspace) = generate_dissimilarity_matrices(points, points_subset);
    //     let oracle = create_oracle_from_data(
    //         dissimilarity_full_space, 
    //         dissimilarity_subspace, 
    //         OrderedFloat(2.5), 
    //         None
    //     ); 
    //     let keys = oracle.get_key_list();
    //     let subcomplex_order_operator = oracle.order_operator_sub_complex_ref().to_owned();
    //     let num_simplices = keys.clone().iter().count(); 
    //     println!(" =============== ");
    //     println!("- The full complex has {:?} simplices.", num_simplices); 
    //     let num_simplices_subcomplex = keys.clone().iter().filter_map(|x| subcomplex_order_operator.diameter(&x.vertices)).count();
    //     println!("- The subcomplex has {:?} simplices.", num_simplices_subcomplex); 

    //     // factor oracle and create the oracle wrapper (this gives us access to methods to compute relative homology!)
    //     let start_1 = std::time::Instant::now(); 
    //     let oracle_factored = oracle.factor_from_arc(); 
    //     let time_to_factor_1 = start_1.elapsed().as_secs_f64();
    //     println!("- It took {:?} seconds to compute the first U-match", time_to_factor_1); 
    //     let oracle_wrapper = RelativeBoundaryMatrixOracleWrapper::new(oracle.clone(), oracle_factored);
        
    //     // get matrices we need from the first U-match
    //     let matching = oracle_wrapper.umatch.matching_ref(); 
    //     let source = oracle_wrapper.umatch.comb_domain(); 
    //     let boundary = oracle_wrapper.umatch.mapping_ref(); 

    //     // perform the second U-match and get the matched basis
    //     let start_2 = std::time::Instant::now();
    //     let comb_product_factored = oracle_wrapper.lazy_comb_product_factored(); 
    //     let time_to_factor_2 = start_2.elapsed().as_secs_f64();
    //     println!("- It took {:?} seconds to compute the second U-match", time_to_factor_2);
    //     let matched_basis = oracle_wrapper.lazy_matched_basis(&comb_product_factored).0;
        
    //     // get some information about homology
    //     let keys_to_generate_filtered_absolute_boundary_basis = matching.filter_only_matched_majors(keys.clone()).collect_vec(); 
    //     let keys_to_generate_filtered_absolute_cycle_basis = matching.filter_out_matched_minors(keys.clone()).collect_vec();
    //     let keys_to_generate_filtered_relative_cycle_basis = oracle_wrapper.filtered_relative_cycle_basis(); 
    //     let keys_to_generate_filtered_relative_boundary_basis = oracle_wrapper.filtered_relative_boundary_basis();
    //     let num_rel_boundaries = keys_to_generate_filtered_relative_boundary_basis.len(); 
        
    //     // iterate over the keys that generate the filtered relative cycle basis
    //     let mut count_rel_boundaries: usize = 0; 
    //     for key in keys_to_generate_filtered_relative_cycle_basis { 
    //         // get the cycle rep 
    //         // NOTE: why do we split into these cases? A few notes here: 
    //         // - in standard PH, certain representatives for subsumed homology classes (boundaries) only appear in the columns of T, while columns of S only contain a basis which include these chains in their span
    //         // - similarly, in this case:
    //         //      (a) certain representatives for essential (relative) homology classes only appear in the columns of S, while columns of AT' simply span them 
    //         //      (b) certain representatives for subsumed (relative) homology classes only appear in the columns of AT', while columns of T simply span them
    //         // - for more, refer to `FactoredBoundaryMatrixVrRelative::persistent_relative_homology()` in `clique_filtered.rs` in the oat_python directory! 
    //         let is_boundary = keys_to_generate_filtered_relative_boundary_basis.contains(&key.clone()); 
    //         let chain = match is_boundary { 
    //             // if T[:,key] is a relative boundary
    //             true => { matched_basis.view_minor_descend(key.clone()).collect_vec() }, 
    //             // if S[:,key] is an essential cycle
    //             false => { source.view_minor_descend(key.clone()).collect_vec() }
    //         };
    //         let length_of_chain = chain.len(); 
    //         let is_chain_in_subcomplex = chain.clone().iter().filter_map(|x| subcomplex_order_operator.diameter(&x.0.vertices)).count() == length_of_chain;
    //         // the boundary of the chain
    //         let boundary_vec = vector_matrix_multiply_minor_descend_simplified(
    //             chain.clone(), 
    //             boundary, 
    //             boundary.ring_operator, 
    //             boundary.row_index_order_operator.clone()
    //         ).collect_vec(); 
    //         let length_of_boundary_vec = boundary_vec.len(); 
    //         let is_boundary_vec_in_subcomplex = boundary_vec.clone().iter().filter_map(|x| subcomplex_order_operator.diameter(&x.0.vertices)).count() == length_of_boundary_vec;
    //         // assert that the boundary of the chain is in the subcomplex 
    //         assert!(is_boundary_vec_in_subcomplex);
    //         // NOTE: since we get representatives for the essential homology classes from S, we only need to check the boundary representatives further! 
    //         if is_boundary { 
    //             count_rel_boundaries = count_rel_boundaries + 1; 
    //             // CASE 1: the boundary is trivial
    //             if length_of_boundary_vec == 0 { 
    //                 // CASE 1.1: ALL absolute boundaries 
    //                 if keys_to_generate_filtered_absolute_boundary_basis.contains(&key) {
    //                     assert!(keys_to_generate_filtered_relative_boundary_basis.contains(&key));
    //                 // CASE 1.2 essential absolute cycle contained ENTIRELY in the subcomplex
    //                 } else {
    //                     if is_chain_in_subcomplex { 
    //                         assert!(keys_to_generate_filtered_absolute_cycle_basis.contains(&key));
    //                         assert!(keys_to_generate_filtered_relative_boundary_basis.contains(&key));
    //                     }
    //                 }
    //             }
    //             // CASE 2: boundary is nontrivial
    //             else { 
    //                 if is_chain_in_subcomplex { // this check is crucial ... chains which are not members of the subcomplex but have a boundary in the subcomplex are NOT relative boundaries!
    //                     assert!(keys_to_generate_filtered_relative_boundary_basis.contains(&key));
    //                     assert!(!keys_to_generate_filtered_absolute_boundary_basis.contains(&key));
    //                 }
    //             }
    //         }  
    //     }
    //     assert_eq!(count_rel_boundaries, num_rel_boundaries);
    //     print!(" =============== ");
    // }

    // #[test]
    // /// 
    // /// We randomly generate data and construct D, the boundary matrix of a filtered, quotient chain complex. 
    // /// 
    // /// Given: a U-match TM = DS and a second U-match T'M' = (A^{-1}B)S' where A and B, respectively, are identical to T and S up to a permutation of columns. 
    // /// 
    // /// Case: a column of the matched basis AT' is indexed by a key `r` such that T[:,r] is an absolute boundary.
    // /// 
    // /// Require: for each such column AT'[:,r] the column vector ST'[:,c] is the associated bounding cycle, where M[r,c] != 0. Thus, D*(ST'[:,c]) - AT'[:,r] is a subcomplex chain. 
    // /// 
    // fn test_computing_bounding_chains_randomized_data() { 
    //     // randomly generate data 
    //     let num_points = 50 as usize; 
    //     let num_subset_points = rand::thread_rng().gen_range((2 as usize)..(num_points)-1);
    //     let (points, points_subset) = generate_random_test_data_on_disk_at_origin(
    //         5.0, 
    //         num_points, 
    //         num_subset_points
    //     ); 

    //     // create dissimilarity matrices and boundary oracle
    //     let (dissimilarity_full_space, dissimilarity_subspace) = generate_dissimilarity_matrices(points, points_subset);
    //     let oracle = create_oracle_from_data(
    //         dissimilarity_full_space, 
    //         dissimilarity_subspace, 
    //         OrderedFloat(2.5), 
    //         None
    //     ); 
    //     let keys = oracle.get_key_list();
    //     let subcomplex_order_operator = oracle.order_operator_sub_complex_ref().to_owned();
    //     let num_simplices = keys.clone().iter().count(); 
    //     println!(" =============== "); 
    //     println!("- The full complex has {:?} simplices.", num_simplices); 
    //     let num_simplices_subcomplex = keys.clone().iter().filter_map(|x| subcomplex_order_operator.diameter(&x.vertices)).count();
    //     println!("- The subcomplex has {:?} simplices.", num_simplices_subcomplex); 
        
    //     // factor oracle and create the oracle wrapper (this gives us access to methods to compute relative homology!)
    //     let start_1 = std::time::Instant::now(); 
    //     let oracle_factored = oracle.factor_from_arc(); 
    //     let time_to_factor_1 = start_1.elapsed().as_secs_f64();
    //     println!("- It took {:?} seconds to compute the first U-match", time_to_factor_1); 
    //     let oracle_wrapper = RelativeBoundaryMatrixOracleWrapper::new(oracle.clone(), oracle_factored);
        
    //     // get information we need from the first U-match
    //     let matching = oracle_wrapper.umatch.matching_ref(); 
    //     let source = oracle_wrapper.umatch.comb_domain(); 
    //     let target = oracle_wrapper.umatch.comb_codomain();
    //     let boundary = oracle_wrapper.umatch.mapping_ref(); 
    //     let ring_operator = oracle_wrapper.umatch.ring_operator();
    //     let source_comb_order_operator = oracle_wrapper.source_comb_order_operator_ref(); 

    //     // get information we need from the second U-match
    //     let start_2 = std::time::Instant::now();
    //     let comb_product_factored = oracle_wrapper.lazy_comb_product_factored(); 
    //     let time_to_factor_2 = start_2.elapsed().as_secs_f64();
    //     println!("- It took {:?} seconds to compute the second U-match", time_to_factor_2);
    //     let (matched_basis, comb_target_of_second_umatch) = oracle_wrapper.lazy_matched_basis(&comb_product_factored);

    //     // iterate over these keys ... 
    //     for r in oracle.get_key_list() { 
    //         if let Some(c) = matching.keymaj_to_keymin(&r) { 
    //             // get cycle representative AT'[:,r]
    //             let mut cycle_representative= vector_matrix_multiply_minor_descend_simplified(
    //                 &comb_target_of_second_umatch.clone().view_minor_descend(r).collect_vec(),
    //                 target.clone(), 
    //                 ring_operator.clone(), 
    //                 source_comb_order_operator.clone(),
    //             ).collect_vec();
    //             // get bounding chain ST'[:,c]
    //             let bounding_chain= vector_matrix_multiply_minor_descend_simplified(
    //                 &comb_target_of_second_umatch.clone().view_minor_descend(c).collect_vec(),
    //                 source.clone(), 
    //                 ring_operator.clone(), 
    //                 source_comb_order_operator.clone(),
    //             ); 
    //             // get the boundary of the bounding chain, or D*(ST'[:,c])
    //             let mut boundary_vec = vector_matrix_multiply_minor_descend_simplified(
    //                 bounding_chain,
    //                 boundary, 
    //                 ring_operator.clone(), 
    //                 source_comb_order_operator.clone(),
    //             ).collect_vec(); 
                
    //             // we do a bit of extra work to ensure that `cycle_representative` and `boundary_vec` are simplified linear combinations and identically ordered by index
    //             cycle_representative.sort_by(|lhs, rhs| oracle_wrapper.mapping.row_index_order_operator.judge_partial_cmp(lhs, rhs).unwrap());
    //             boundary_vec.sort_by(|lhs, rhs| oracle_wrapper.mapping.row_index_order_operator.judge_partial_cmp(lhs, rhs).unwrap());
    //             let boundary_vec_formatted = boundary_vec.clone().into_iter().peekable().simplify(ring_operator).collect_vec();
    //             // ASSERT that D*(ST'[:,c]) - AT'[:,r] is a subcomplex chain for each bounding chain ST'[:,c]
    //             // equivalently, we assert that `boundary_vec_formatted` and `cycle_representative` are identical once we have ensured that each linear combination: 
    //             // - is simplified,
    //             // - identically ordered by index,
    //             // - and contains no subcomplex simplices 
    //             let trimmed_cycle_representative = oracle_wrapper.trim_relative_chain(cycle_representative.clone());
    //             let trimmed_boundary_vec = oracle_wrapper.trim_relative_chain(boundary_vec_formatted.clone()); 
    //             let length_trimmed_cycle = trimmed_cycle_representative.len(); 
    //             let length_trimmed_boundary_vec = trimmed_boundary_vec.len();
    //             assert!(length_trimmed_cycle == length_trimmed_boundary_vec); 
    //             for i in 0..length_trimmed_cycle { 
    //                 // we compare the indices and not the entries (ie the chains are identical up to orientation!)
    //                 assert_eq!(trimmed_cycle_representative[i].0, trimmed_boundary_vec[i].0);
    //             }
    //         }
    //     }
    //     println!(" =============== "); 
    // }

    // NOTE for developers : the code required for this unit test is still in development ... see oat_rust/src/algebra/chains/relative/oracle.rs
    // #[test]
    // ///  
    // /// We randomly generate data and construct D, the boundary matrix of a filtered, quotient chain complex. 
    // /// 
    // /// Case: extracting a persistence moudle using the keys returned from `RelativeBoundaryMatrixOracleWrapper::filtered_relative_cycle_basis`
    // /// 
    // /// Require: the matched basis contains all boundary representatives, and spans all relative cycles and relative boundaries. 
    // /// 
    // /// NOTE: we compute the matched basis via a sparse, and explicitly sorted COMB product A^{-1}B
    // /// 
    // fn test_computing_sparse_matched_basis_matrix_randomized_data() { 
    //     // randomly generate data 
    //     let num_points = 11 as usize; 
    //     let num_subset_points = rand::thread_rng().gen_range((2 as usize)..(num_points)-1);
    //     let (points, points_subset) = generate_random_test_data_on_disk_at_origin(
    //         5.0, 
    //         num_points, 
    //         num_subset_points
    //     ); 

    //     println!(" =============== "); 

    //     // for (randomly) testing the custom filtration 
    //     let diss_max = OrderedFloat(2.5);
    //     let mut diss_max_custom = 0.0; 
    //     let custom_subcomplex_data;
    //     let mut rng = rand::thread_rng();
    //     let probability: f64 = rng.gen(); 
    //     // CASE: customize the subcomplex filtration 
    //     if probability > 0.2 {  
    //         let mut custom_subcomplex_data_inner = Vec::new(); 
    //         for i in 0..num_points { 
    //             if i < num_subset_points - 1 {
    //                 // check that we have a point in the subset 
    //                 assert!(points[i] == points_subset[i]); 
    //                 // randomly generate a float 
    //                 let float: f64 = rng.gen();
    //                 // customize its filtration 
    //                 let custom_filtration = diss_max.into_inner() - float; 
    //                 if custom_filtration > diss_max_custom { 
    //                     diss_max_custom = custom_filtration; 
    //                 }
    //                 custom_subcomplex_data_inner.push( Some(OrderedFloat(custom_filtration)) );
    //             }
    //             else { 
    //                 // if we do not have a subcomplex point, then we do nothing!
    //                 custom_subcomplex_data_inner.push( None );
    //             }
    //         }
    //         custom_subcomplex_data = Some(custom_subcomplex_data_inner); 
    //         println!("- We customized the subcomplex data");
    //         println!("- The largest custom diameter is {:?}", diss_max_custom);
    //     // CASE: do not customize the subcomplex data
    //     } else { 
    //         custom_subcomplex_data = None; 
    //     }

    //     // create dissimilarity matrices and boundary oracle
    //     let (dissimilarity_full_space, dissimilarity_subspace) = generate_dissimilarity_matrices(points, points_subset);
    //     let oracle = create_oracle_from_data(
    //         dissimilarity_full_space, 
    //         dissimilarity_subspace, 
    //         diss_max, 
    //         custom_subcomplex_data
    //     ); 
    //     let keys = oracle.get_key_list();
    //     let subcomplex_order_operator = oracle.order_operator_sub_complex_ref().to_owned();
    //     let num_simplices = keys.clone().iter().count(); 
    //     println!("- The full complex has {:?} simplices.", num_simplices); 
    //     let num_simplices_subcomplex = keys.clone().iter().filter_map(|x| subcomplex_order_operator.diameter(&x.vertices)).count();
    //     println!("- The subcomplex has {:?} simplices.", num_simplices_subcomplex); 

    //     // factor oracle and create the oracle wrapper (this gives us access to methods to compute relative homology!)
    //     let start_1 = std::time::Instant::now(); 
    //     let oracle_factored = oracle.factor_from_arc(); 
    //     let time_to_factor_1 = start_1.elapsed().as_secs_f64();
    //     println!("- It took {:?} seconds to compute the first U-match", time_to_factor_1); 
    //     let oracle_wrapper = RelativeBoundaryMatrixOracleWrapper::new(oracle.clone(), oracle_factored);
        
    //     // get matrices we need from the first U-match
    //     let matching = oracle_wrapper.umatch.matching_ref(); 
    //     let source = oracle_wrapper.umatch.comb_domain(); 
    //     let boundary = oracle_wrapper.umatch.mapping_ref(); 

    //     // perform the second U-match and get the matched basis
    //     let start_2 = std::time::Instant::now();
    //     let comb_product_factored = oracle_wrapper.sparse_comb_product_factored(); 
    //     let time_to_factor_2 = start_2.elapsed().as_secs_f64();
    //     println!("- It took {:?} seconds to compute the second U-match", time_to_factor_2);
    //     let matched_basis = oracle_wrapper.sparse_matched_basis(&comb_product_factored).0;
        
    //     // get some information about homology
    //     let keys_to_generate_filtered_absolute_boundary_basis = matching.filter_only_matched_majors(keys.clone()).collect_vec(); 
    //     let keys_to_generate_filtered_absolute_cycle_basis = matching.filter_out_matched_minors(keys.clone()).collect_vec();
    //     let keys_to_generate_filtered_relative_cycle_basis = oracle_wrapper.filtered_relative_cycle_basis(); 
    //     let keys_to_generate_filtered_relative_boundary_basis = oracle_wrapper.filtered_relative_boundary_basis();
    //     let num_rel_boundaries = keys_to_generate_filtered_relative_boundary_basis.len(); 
        
    //     // iterate over the keys that generate the filtered relative cycle basis
    //     let mut count_rel_boundaries: usize = 0; 
    //     for key in keys_to_generate_filtered_relative_cycle_basis { 
    //         // get the cycle rep 
    //         // NOTE: why do we split into these cases? A few notes here: 
    //         // - in standard PH, certain representatives for subsumed homology classes (boundaries) only appear in the columns of T, while columns of S only contain a basis which include these chains in their span
    //         // - similarly, in this case:
    //         //      (a) certain representatives for essential (relative) homology classes only appear in the columns of S, while columns of AT' simply span them 
    //         //      (b) certain representatives for subsumed (relative) homology classes only appear in the columns of AT', while columns of T simply span them
    //         // - for more, refer to `FactoredBoundaryMatrixVrRelative::persistent_relative_homology()` in `clique_filtered.rs` in the oat_python directory! 
    //         let is_boundary = keys_to_generate_filtered_relative_boundary_basis.contains(&key.clone()); 
    //         let chain = match is_boundary { 
    //             // if T[:,key] is a relative boundary
    //             true => { 
    //                 println!("- Getting the cycle rep from the matched basis.");
    //                 matched_basis.view_minor_descend(key.clone()).collect_vec() 
    //             }, 
    //             // if S[:,key] is an essential cycle
    //             false => { 
    //                 println!("- Getting the cycle rep from the original source COMB.");
    //                 source.view_minor_descend(key.clone()).collect_vec() 
    //             }
    //         };
    //         let length_of_chain = chain.len(); 
    //         let is_chain_in_subcomplex = chain.clone().iter().filter_map(|x| subcomplex_order_operator.diameter(&x.0.vertices)).count() == length_of_chain;
    //         // the boundary of the chain
    //         let boundary_vec = vector_matrix_multiply_minor_descend_simplified(
    //             chain.clone(), 
    //             boundary, 
    //             boundary.ring_operator, 
    //             boundary.row_index_order_operator.clone()
    //         ).collect_vec(); 
    //         let length_of_boundary_vec = boundary_vec.len(); 
    //         let is_boundary_vec_in_subcomplex = boundary_vec.clone().iter().filter_map(|x| subcomplex_order_operator.diameter(&x.0.vertices)).count() == length_of_boundary_vec;
    //         // assert that, if the cycle representative is incorrect (and thus not in the matched basis), then it is a trivial class
    //         if !is_boundary_vec_in_subcomplex {
    //             let absolute_cycle_birth = oracle.column_index_order_operator.diameter(&key.vertices).unwrap();
    //             let relative_cycle_birth = oracle_wrapper.source_comb_order_operator.relative_cycle_birth(&key).unwrap();
    //             let absolute_boundary_birth = oracle.column_index_order_operator.diameter(&matching.keymaj_to_keymin(&key).unwrap().vertices).unwrap();
    //             let relative_boundary_birth = oracle_wrapper.target_comb_order_operator.relative_boundary_birth(&key).unwrap();
    //             assert!(absolute_cycle_birth == relative_cycle_birth);
    //             assert!(relative_cycle_birth == absolute_boundary_birth);
    //             assert!(absolute_boundary_birth == relative_boundary_birth);
    //         }

    //         // NOTE: since we get representatives for the essential homology classes from S, we only need to check the boundary representatives further! 
    //         if is_boundary { 
    //             count_rel_boundaries = count_rel_boundaries + 1; 
    //             // CASE 1: the boundary is trivial
    //             if length_of_boundary_vec == 0 { 
    //                 // CASE 1.1: ALL absolute boundaries 
    //                 if keys_to_generate_filtered_absolute_boundary_basis.contains(&key) {
    //                     assert!(keys_to_generate_filtered_relative_boundary_basis.contains(&key));
    //                 // CASE 1.2 essential absolute cycle contained ENTIRELY in the subcomplex
    //                 } else {
    //                     if is_chain_in_subcomplex { 
    //                         assert!(keys_to_generate_filtered_absolute_cycle_basis.contains(&key));
    //                         assert!(keys_to_generate_filtered_relative_boundary_basis.contains(&key));
    //                     }
    //                 }
    //             }
    //             // CASE 2: boundary is nontrivial
    //             else { 
    //                 if is_chain_in_subcomplex { // this check is crucial ... chains which are not members of the subcomplex but have a boundary in the subcomplex are NOT relative boundaries!
    //                     assert!(keys_to_generate_filtered_relative_boundary_basis.contains(&key));
    //                     assert!(!keys_to_generate_filtered_absolute_boundary_basis.contains(&key));
    //                 }
    //             }
    //         }  
    //     }
    //     assert_eq!(count_rel_boundaries, num_rel_boundaries);
    //     print!(" =============== ");
    // }