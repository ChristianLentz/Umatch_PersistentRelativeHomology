
//! 
//! This module equips Rust's `CsMatBase` structure with the essential matrix traits that make it usable within the OAT ecosystem. 
//! 

// Import crates 
use crate::algebra::matrices::query::{IndicesAndCoefficients, MatrixEntry, ViewColDescend, ViewRowAscend};
use sprs::{CsMatBase, TriMatBase};
use std::collections::HashMap;
use std::fmt::Debug;
use std::ops::{Add, Deref}; 
use std::hash::Hash;

#[derive(Clone, Debug)]
///
/// A sparse matrix oracle which wraps a [`CsMatBase`] struct from the [`sprs`] crate. This struct 
/// implements necessary oracle traits which make the struct usable within the OAT ecosystem. 
/// 
pub struct CompressedSparse<N, I, IptrStorage, IndStorage, DataStorage, RowIdx, ColIdx>
    where
        N: Clone + Debug + Default,
        I: sprs::SpIndex,
        IptrStorage: AsRef<[I]> + Debug + Deref<Target = [I]>,
        IndStorage: AsRef<[I]> + Debug + Deref<Target = [I]>,
        DataStorage: AsRef<[N]> + Debug + Deref<Target = [N]>,
        RowIdx: Clone + Eq + Hash, 
        ColIdx: Clone + Eq + Hash 
{
    /// The wrapped [`CsMatBase`] in CSR format 
    inner_csr: CsMatBase<N, I, IptrStorage, IndStorage, DataStorage>,
    /// The wrapped [`CsMatBase`] in CSC format 
    inner_csc: CsMatBase<N, I, Vec<I>, Vec<I>, Vec<N>>,
    /// A [`HashMap`] between the sorted row indices of Self and the integers
    hash_map_row_indices: HashMap<RowIdx, usize>, 
    /// A copy of the sorted row indices.
    sorted_row_indices: Vec<RowIdx>, 
    /// A [`HashMap`] between the sorted column indices of Self and the integers
    hash_map_column_indices: HashMap<ColIdx, usize>, 
    /// A copy of the sorted column indices.
    sorted_column_indices: Vec<ColIdx>
}

impl <N, I, IptrStorage, IndStorage, DataStorage, RowIdx, ColIdx> 
    CompressedSparse<N, I, IptrStorage, IndStorage, DataStorage, RowIdx, ColIdx> 
        where 
            N: Clone + Debug + Default,
            I: sprs::SpIndex,
            IptrStorage: AsRef<[I]> + Debug + Deref<Target = [I]>,
            IndStorage: AsRef<[I]> + Debug + Deref<Target = [I]>,
            DataStorage: AsRef<[N]> + Debug + Deref<Target = [N]>,
            RowIdx: Clone + Eq + Hash, 
            ColIdx: Clone + Eq + Hash 
{ 

    /// 
    /// Construct a new [`CompressedSparse`] matrix from a [`CsMatBase`] in CSR format. 
    /// 
    /// It is assumed that the rows and columns of the provided matrix are indexed by integers in 
    /// ascending order (for example, an n x n matrix will have rows and columns identically indexed 
    /// by the set {1, ..., n}). However, the user must also provide SORTED lists of row and column 
    /// indices. This constructor creates hash maps between these lists and the integers and uses 
    /// this bijection to map the indices of the wrapped sparse matrix to the indices of `Self`. 
    /// 
    /// For example, suppose that R is the vector of row indices of a wrapped CSR matrix M and h is 
    /// a hash map h: R -> Z where Z is the integers. If r in R then the row of `Self` indexed by r is 
    /// the row of M indexed by the integer h(r). 
    /// 
    /// This hashing construction allows the [`CompressedSparse`] oracle to be compatible with row and 
    /// column indices that differe in generic type and/or order. 
    /// 
    pub fn new(
        inner: CsMatBase<N, I, IptrStorage, IndStorage, DataStorage>, 
        sorted_row_indices: Vec<RowIdx>, 
        sorted_column_indices: Vec<ColIdx>
    ) -> Self {
        // map the CSR to a CSC format 
        let inner_csc = inner.to_other_storage(); 
        // create hash maps from generic index types to integers
        let hash_map_row_indices: HashMap<RowIdx, usize> = sorted_row_indices
            .iter()
            .enumerate()
            .map(|(i, t)| (t.clone(), i)) 
            .collect();
        let hash_map_column_indices: HashMap<ColIdx, usize> = sorted_column_indices
            .iter()
            .enumerate()
            .map(|(i, t)| (t.clone(), i)) 
            .collect();
        // return the instance of self 
        Self { 
            inner_csr: inner, 
            inner_csc, 
            hash_map_row_indices, 
            sorted_row_indices: sorted_row_indices.clone(),
            hash_map_column_indices, 
            sorted_column_indices: sorted_column_indices.clone()
        }
    }

    /// 
    /// Return a reference to the inner [`CsMatBase`] matrix in CSR format.
    /// 
    pub fn inner_csr(&self) -> &CsMatBase<N, I, IptrStorage, IndStorage, DataStorage> {
        &self.inner_csr
    }

    /// 
    /// Return a reference to the inner [`CsMatBase`] matrix in CSC format.
    /// 
    pub fn inner_csc(&self) -> &CsMatBase<N, I, Vec<I>, Vec<I>, Vec<N>> {
        &self.inner_csc
    }

    /// 
    /// Return a mutable reference to the inner [`CsMatBase`] in CSR format.
    /// 
    pub fn inner_mut_csr(&mut self) -> &mut CsMatBase<N, I, IptrStorage, IndStorage, DataStorage> {
        &mut self.inner_csr
    }

    /// 
    /// Return a mutable reference to the inner [`CsMatBase`] in CSC cformat.
    /// 
    pub fn inner_mut_csc(&mut self) -> &mut CsMatBase<N, I, Vec<I>, Vec<I>, Vec<N>> {
        &mut self.inner_csc
    }

    ///
    /// Get the shape of the inner CSR matrix. 
    /// 
    pub fn shape_csr(&self) -> (usize, usize) {
        self.inner_csr.shape()
    }

    ///
    /// Get the shape of the inner CSC matrix. 
    /// 
    pub fn shape_csc(&self) -> (usize, usize) {
        self.inner_csc.shape()
    }

}

// Implement `IndicesAndCoefficients`
impl <N, I, IptrStorage, IndStorage, DataStorage, RowIdx, ColIdx> 
    IndicesAndCoefficients
        for CompressedSparse<N, I, IptrStorage, IndStorage, DataStorage, RowIdx, ColIdx> 
            where 
                N: Clone + Debug + Default,
                I: sprs::SpIndex,
                IptrStorage: AsRef<[I]> + Debug + Deref<Target = [I]>,
                IndStorage: AsRef<[I]> + Debug + Deref<Target = [I]>,
                DataStorage: AsRef<[N]> + Debug + Deref<Target = [N]>,
                RowIdx: Clone + Eq + Hash, 
                ColIdx: Clone + Eq + Hash 
{ 
    type EntryMajor = (Self::ColIndex, Self::Coefficient);
    type EntryMinor = (Self::RowIndex, Self::Coefficient);    
    type RowIndex = RowIdx; 
    type ColIndex = ColIdx; 
    type Coefficient = N;
}

// Implement `ViewRowAscend`
impl <N, I, IptrStorage, IndStorage, DataStorage, RowIdx, ColIdx> 
    ViewRowAscend
        for CompressedSparse<N, I, IptrStorage, IndStorage, DataStorage, RowIdx, ColIdx> 
            where 
                N: Clone + Debug + Default,
                I: sprs::SpIndex,
                IptrStorage: AsRef<[I]> + Debug + Deref<Target = [I]>,
                IndStorage: AsRef<[I]> + Debug + Deref<Target = [I]>,
                DataStorage: AsRef<[N]> + Debug + Deref<Target = [N]>,
                RowIdx: Clone + Eq + Hash + Debug, 
                ColIdx: Clone + Eq + Hash 
{ 
    type ViewMajorAscend = Vec<Self::EntryMajor>;
    type ViewMajorAscendIntoIter = std::vec::IntoIter<Self::EntryMajor>;
    /// 
    /// Get a major (row) view of `CompressedSparse`.
    /// 
    fn view_major_ascend(&self, keymaj: Self::RowIndex) -> Self::ViewMajorAscend {
        let i = self.hash_map_row_indices[&keymaj]; 
        if let Some(row) = self.inner_csr.outer_view(i) {
            row.iter().map(|(j, v)| (self.sorted_column_indices[j].clone(), v.clone())).collect()
        } else { 
            panic!("\n\nError: Index out of bounds when retreiving row {:?} of `CompressedSparse` matrix struct. \n This message is generated by OAT.\n\n", keymaj);

        }
    }
}

// Implement `ViewColDescend`
impl <N, I, IptrStorage, IndStorage, DataStorage, RowIdx, ColIdx> 
    ViewColDescend
        for CompressedSparse<N, I, IptrStorage, IndStorage, DataStorage, RowIdx, ColIdx> 
            where 
                N: Clone + Debug + Default,
                I: sprs::SpIndex,
                IptrStorage: AsRef<[I]> + Debug + Deref<Target = [I]>,
                IndStorage: AsRef<[I]> + Debug + Deref<Target = [I]>,
                DataStorage: AsRef<[N]> + Debug + Deref<Target = [N]>,
                RowIdx: Clone + Eq + Hash, 
                ColIdx: Clone + Eq + Hash + Debug
{ 
    type ViewMinorDescend = Vec<Self::EntryMinor>;
    type ViewMinorDescendIntoIter = std::vec::IntoIter<Self::EntryMinor>;
    /// 
    /// Get a minor (column) view of `CompressedSparse`.
    /// 
    fn view_minor_descend(&self, keymin: Self::ColIndex) -> Self::ViewMinorDescend {
        let j = self.hash_map_column_indices[&keymin]; 
        if let Some(col) = self.inner_csc.outer_view(j) {
            col.iter().map(|(i, v)| (self.sorted_row_indices[i].clone(), v.clone())).collect()
        } else { 
            panic!("\n\nError: Index out of bounds when retreiving column {:?} of `CompressedSparse` matrix struct. \n This message is generated by OAT.\n\n", keymin);

        }
    }
}

// Implement `MatrixEntry`
impl <N, I, IptrStorage, IndStorage, DataStorage, RowIdx, ColIdx> 
    MatrixEntry
        for CompressedSparse<N, I, IptrStorage, IndStorage, DataStorage, RowIdx, ColIdx> 
            where 
                N: Clone + Debug + Default,
                I: sprs::SpIndex,
                IptrStorage: AsRef<[I]> + Debug + Deref<Target = [I]>,
                IndStorage: AsRef<[I]> + Debug + Deref<Target = [I]>,
                DataStorage: AsRef<[N]> + Debug + Deref<Target = [N]>,
                RowIdx: Clone + Eq + Hash, 
                ColIdx: Clone + Eq + Hash 
{ 
    /// 
    /// Get entry (keymaj, keymin) of the wrapped sparse matrix. 
    /// 
    fn entry_major_at_minor( &self, keymaj: Self::RowIndex, keymin: Self::ColIndex, ) -> Option< Self::Coefficient > {
        if let Some(entry) = self.inner_csr.get(self.hash_map_row_indices[&keymaj], self.hash_map_column_indices[&keymin]) { 
            Some(entry.clone())
        } else { 
            None
        }
    }
}

///
/// Constrct a CSR matrix from a lazy (row-major) matrix oracle. This is used as an intermediate/helper to construct the 
/// [`CompressedSparse`] struct. 
/// 
pub fn lazy_oracle_to_csr_view_major<Matrix, Entry, RowIndex, ColIndex>(
    matrix: Matrix, 
    row_indices: Vec<RowIndex>, 
    col_indices: Vec<ColIndex>
) -> CsMatBase<Entry, usize, Vec<usize>, Vec<usize>, Vec<Entry>>

where 
    Matrix: ViewRowAscend + IndicesAndCoefficients<Coefficient=Entry, RowIndex=RowIndex, ColIndex=ColIndex, EntryMajor=(ColIndex,Entry), EntryMinor=(RowIndex,Entry)>, 
    RowIndex: Clone + Eq + Hash, 
    ColIndex: Clone + Eq + Hash, 
    Entry: Clone + Add<Output=Entry>
{ 
    let shape = (row_indices.len(), col_indices.len()); 

    // assign a bijection from keys to integers for both the rows and columns
    // NOTE: COMBs inherit the keys (including order) of the factored oracle
    let row_bijection: HashMap<_,_> = row_indices.iter().cloned().enumerate().map(|(x,y)| (y,x) ).collect();
    let col_bijection: HashMap<_,_> = col_indices.iter().cloned().enumerate().map(|(x,y)| (y,x) ).collect();

    // format data from the matrix to export ... collect it in the vectors below
    let mut indices_row = Vec::new();
    let mut indices_col = Vec::new();
    let mut vals: Vec<Entry> = Vec::new();
    for row_index in row_indices.iter().cloned() {
        for (col_index, coefficient) in matrix.view_major_ascend(row_index.clone()) {
            indices_row.push(row_bijection[&row_index.clone()].clone()); 
            indices_col.push(col_bijection[&col_index.clone()].clone()); 
            vals.push(coefficient); 
        }
    }
    TriMatBase::from_triplets(shape, indices_row, indices_col, vals).to_csr()
}

///
/// Constrct a CSR matrix from a lazy (column-major) matrix oracle. This is used as an intermediate/helper to construct the 
/// [`CompressedSparse`] struct. 
/// 
pub fn lazy_oracle_to_csr_view_minor<Matrix, Entry, RowIndex, ColIndex>(
    matrix: Matrix, 
    row_indices: Vec<RowIndex>, 
    col_indices: Vec<ColIndex>
) -> CsMatBase<Entry, usize, Vec<usize>, Vec<usize>, Vec<Entry>>

where 
    Matrix: ViewColDescend + IndicesAndCoefficients<Coefficient=Entry, RowIndex=RowIndex, ColIndex=ColIndex, EntryMajor=(ColIndex,Entry), EntryMinor=(RowIndex,Entry)>, 
    RowIndex: Clone + Eq + Hash, 
    ColIndex: Clone + Eq + Hash, 
    Entry: Clone + Add<Output=Entry> 
{ 
    let shape = (row_indices.len(), col_indices.len()); 

    // assign a bijection from keys to integers for both the rows and columns
    // NOTE: COMBs inherit the keys (including order) of the factored oracle
    let row_bijection: HashMap<_,_> = row_indices.iter().cloned().enumerate().map(|(x,y)| (y,x) ).collect();
    let col_bijection: HashMap<_,_> = col_indices.iter().cloned().enumerate().map(|(x,y)| (y,x) ).collect();

    // format data from the matrix to export ... collect it in the vectors below
    let mut indices_row = Vec::new();
    let mut indices_col = Vec::new();
    let mut vals: Vec<Entry> = Vec::new();
    for col_index in col_indices.iter().cloned() {
        for (row_index, coefficient) in matrix.view_minor_descend(col_index.clone()) {
            indices_row.push(row_bijection[&row_index.clone()].clone()); 
            indices_col.push(col_bijection[&col_index.clone()].clone()); 
            vals.push(coefficient); 
        }
    }
    TriMatBase::from_triplets(shape, indices_row, indices_col, vals).to_csr()
}

#[cfg(test)]

mod tests {

    use sprs::CsMatBase;
    use crate::algebra::matrices::{query::MatrixEntry, types::compressed_sparse::CompressedSparse};

    #[test]
    pub fn test_sparse_matrix_construction() { 
    
        // The matrix used for this unit test
        // [1.0 0.0 2.0 0.0]
        // [0.0 3.0 0.0 4.0]
        // [5.0 0.0 6.0 0.0]

        // row pointer
        let indptr = vec![0, 2, 4, 6];
        // column indices of each nonzero entry
        let indices = vec![0, 2, 1, 3, 0, 2];
        // nonzero values in row-major order
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

        // create matrix 
        let csr: CsMatBase<f64, usize, Vec<usize>, Vec<usize>, Vec<f64>> = CsMatBase::new((3, 4), indptr.clone(), indices.clone(), data.clone());

        // create an instance of `CompressedSparse` 
        let test_obj = CompressedSparse::new(
            csr.clone(), 
            vec![0,1,2], 
            vec![0,1,2,3]
        ); 

        // make sure it properly wraps the sparse matrix
        for (outer_idx, vec) in csr.outer_iterator().enumerate() {
            for (inner_idx, val) in vec.iter() {
                assert_eq!(val.clone(), test_obj.entry_major_at_minor(outer_idx, inner_idx).unwrap()); 
            }
        }
    }

    #[test]
    pub fn test_sparse_matrix_sorting() { 
    
        // The matrix used for this unit test: 
        // [1.0 0.0 2.0 0.0]
        // [0.0 3.0 0.0 4.0]
        // [5.0 0.0 6.0 0.0]

        // row pointer
        let indptr1 = vec![0, 2, 4, 6];
        // column indices of each nonzero entry
        let indices1 = vec![0, 2, 1, 3, 0, 2];
        // nonzero values in row-major order
        let data1 = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        // create matrix 
        let csr1: CsMatBase<f64, usize, Vec<usize>, Vec<usize>, Vec<f64>> = CsMatBase::new((3, 4), indptr1.clone(), indices1.clone(), data1.clone());

        // Create an instance of `CompressedSparse`.
        // Notice the transpositions swapping:
        // - row 1 and row 2
        // - column 0 and column 1
        // This is encoded in the sorted indices that we provide! 
        let permuted_csr1 = CompressedSparse::new(
            csr1.clone(), 
            vec![0,2,1], 
            vec![1,0,2,3]
        ); 

        // The matrix that we have constructed via `CompressedSparse`: 
        // [0.0 1.0 2.0 0.0]
        // [0.0 5.0 6.0 0.0]
        // [3.0 0.0 0.0 4.0]

        // Now let's store this as a `CsMatBase` and compare the two 
        // row pointer
        let indptr2 = vec![0, 2, 4, 6];
        // column indices of each nonzero entry
        let indices2 = vec![1, 2, 1, 2, 0, 3];
        // nonzero values in row-major order
        let data2 = vec![1.0, 2.0, 5.0, 6.0, 3.0, 4.0];
        // create matrix 
        let csr2: CsMatBase<f64, usize, Vec<usize>, Vec<usize>, Vec<f64>> = CsMatBase::new((3, 4), indptr2.clone(), indices2.clone(), data2.clone());

        // assert that the two matrices are the same
        for (outer_idx, vec) in csr2.outer_iterator().enumerate() {
            for (inner_idx, val) in vec.iter() {
                assert_eq!(val.clone(), permuted_csr1.entry_major_at_minor(outer_idx, inner_idx).unwrap()); 
            }
        }
    }
}