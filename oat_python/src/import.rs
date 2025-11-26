//! Import data from Python

use itertools::Itertools;
use ordered_float::OrderedFloat;
use pyo3::pyfunction;
use pyo3::prelude::*;
use pyo3::types::PyType;

use sprs::CsMatBase;


pub fn import_sparse_matrix(py: Python, scipy_csr: &PyAny )     
    -> PyResult< CsMatBase<OrderedFloat<f64>, usize, Vec<usize>, Vec<usize>, Vec<OrderedFloat<f64>>> >    
{
    // Check if the object is an instance of csr_matrix and unpack its attributes
    let shape: (usize,usize) = scipy_csr.getattr("shape").ok().unwrap().extract().ok().unwrap();   
    let indptr: Vec<usize> = scipy_csr.getattr("indptr").ok().unwrap().extract().ok().unwrap();  
    let mut indices: Vec<usize> = scipy_csr.getattr("indices").ok().unwrap().extract().ok().unwrap();
    let data: Vec< f64 > = scipy_csr.getattr("data").ok().unwrap().extract().ok().unwrap();
    let mut data = data.into_iter().map(|v| OrderedFloat(v)).collect_vec();

    // NOTES: 
    // - the `indices` array above gives the column indices of each nonzero element of each row of the SciPy sparse CSR matrix, flattened to a 1D vector.
    // - the `indptr` array gives pointers (or integer indices) into the `indices` array which describe the start/end of each new row
    // - Rust WILL panic if the indices array is not sorted: ascending by row with ascending column indices within each row
    
    // EXAMPLE: 
    // consider the 3 x 4 matrix whose rows are: 
    // { 0, 10, 20, 0 }
    // { 0,  0, 30, 0 }
    // { 40, 0, 0 , 50}
    // here are its CSR attributes, properly formatted for Rust: 
    // data = {10, 20, 30, 40, 50}
    // indices = {1,2,2,0,3}
    // indptr = {0,2,3,5} 
    
    // IMPORTANT: sometimes we need to check and explicitly sort the `indices` array, which also requires sorting the `data` array.
    if !are_indices_sorted(&indptr, &indices) { 
        sort_csr_indices(&indptr, &mut indices, &mut data); 
    }

    // return the unpacked CSR matrix
    return Ok( CsMatBase::new(
        shape, // shape: 
        indptr,
        indices,
        data,
    ) )
}

///
/// A function to check if the `indices` array of a SciPy CSR matrix imported from python is sorted. 
/// 
fn are_indices_sorted(
    indptr: &Vec<usize>, 
    indices: &Vec<usize>
) -> bool

{ 
    for row in 0..indptr.len() - 2 { 
        // for each pair of (start, end) indices in the `indptr` array ... 
        let start = indptr[row]; 
        let end = indptr[row + 1]; 
        if start < end { 
            // if the pair is valid, get all column indices for the associated row 
            let row_indices = &indices[start..end]; 
            // assert that each contiguous pair of column indices is sorted 
            if row_indices.windows(2).any(|w| w[0] > w[1]) {
                return false;
            }
        }
    }
    true
} 

///
/// A function to sort the `indices` array of a SciPy CSR matrix, in the case that it is not sorted upon import from Python. 
/// 
fn sort_csr_indices(
    indptr: &Vec<usize>,
    indices: &mut Vec<usize>,
    data: &mut Vec<OrderedFloat<f64>>,
) {
    for row in 0..indptr.len() - 1 {
        // for each pair of (start, end) indices in the `indptr` array
        let start = indptr[row];
        let end = indptr[row + 1];

        // get an iterator over the column `indices` which constitute the row associated with the (start, end) pair
        let row_indices = indices[start..end].iter().cloned(); 
        // get an iterator over the data stored at the column `indices` of this row
        let row_data = data[start..end].iter().cloned(); 

        // now we zip the iterators into a single iterator over (col_index, entry) pairs of the row
        // NOTE: this is the iterator that we sort, and we sort it by the `col_index` key. 
        // NOTE: sorting only the indices and not the associated data would corrupt the matrix! 
        let mut sorted_pairs: Vec<(usize, OrderedFloat<f64>)> = row_indices.zip(row_data).collect(); 
        sorted_pairs.sort_by_key(|&(col, _)| col);

        // once sorted, we mutate our references to `data` and indices`
        for i in 0..sorted_pairs.len() {
            indices[start + i] = sorted_pairs[i].0;
            data[start + i] = sorted_pairs[i].1;
        }
    }
}