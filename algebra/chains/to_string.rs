//!
//! A trait for formatting a chain of simplices as a String. This is introduced primarily for use with exporting data to Python via Pyo3. 
//! 
//! This trait is compatible with chains of the form `Vec< (Simplexfiltered<FilVal>, Coefficient) >`, where coefficient implements the 
//! `Display` trait, thus automatically implementing the `ToString` trait. 
//! 

use crate::algebra::rings::operator_traits::{DivisionRing, Ring, Semiring}; 
use crate::topology::simplicial::simplices::filtered::SimplexFiltered;
use std::fmt::{Debug, Display}; 

//  ===========================================================
//  CHAIN TO STRING TRAIT
//  ===========================================================

/// A trait for formatting a chain of simplices as a String. This is introduced primarily for use with exporting data to Python via Pyo3. 
///
/// This trait is compatible with chains of the form `Vec< (Simplexfiltered<FilVal>, Coefficient) >`, where coefficient implements the 
/// `Display` trait, thus automatically implementing the `ToString` trait. 
pub trait ChainToString<FilVal, Coefficient, RingOperator> 
    where 
        FilVal: Clone + Debug, 
        Coefficient: Clone + Debug + Display + PartialOrd, 
        RingOperator: Semiring<Coefficient> + Ring<Coefficient> + DivisionRing<Coefficient> + Copy + Debug
{ 
    ///
    /// Wrap a chain of simplices to a string. To format the string properly, including the coefficients of the linear combination, 
    /// the user must provide a ring operator. 
    /// 
    fn chain_to_string(&self, ring_operator: RingOperator) -> String; 
}

//  ===========================================================
//  IMPL FOR CHAINS OF FILTERED SIMPLICES
//  ===========================================================

impl <FilVal, Coefficient, RingOperator>
    ChainToString<FilVal, Coefficient, RingOperator> for 
        Vec< (SimplexFiltered<FilVal>, Coefficient) >
            where 
                FilVal: Clone + Debug, 
                Coefficient: Clone + Debug + Display + PartialOrd, 
                RingOperator: Semiring<Coefficient> + Ring<Coefficient> + DivisionRing<Coefficient> + Copy + Debug
{ 
    ///
    /// Wrap a chain of simplices to a string. To format the string properly, including the coefficients of the linear combination, 
    /// the user must provide a ring operator. Note that we assume the linear combination is already simplified, and thus we do 
    /// not check for dropping terms whose coefficient is the additive identity. 
    /// 
    fn chain_to_string(&self, ring_operator: RingOperator) -> String {
        let len = self.clone().iter().count(); 
        let mut linear_combination: String = String::from(""); 
        // for each term of the view/linear combination
        for j in 0..len { 
            let entry: (SimplexFiltered<FilVal>, Coefficient) = self[j].clone();
            // format the simplex as a string 
            let simplex_as_string = entry.0.vertices_to_string(); 
            // format the coefficient as a string: 
            let coef = entry.1.clone(); 
            let coefficient_as_string: String; 
            // NOTE: drop coefficient if it is the multiplicative identity of the ring 
            if ring_operator.is_1(coef.clone()) { 
                if j == 0 { 
                    coefficient_as_string = String::from(""); 
                } else { 
                    coefficient_as_string = String::from("+"); 
                }
            }
            // NOTE: drop coefficient (but keep sign) if it is the additive inverse of the multiplicative identity
            else if ring_operator.is_1(ring_operator.negate(coef.clone())) { 
                coefficient_as_string = String::from("-"); 
            }
            // NOTE: if coefficient is positive (and its not the first term of the liner combination), we need to insert an addition sign
            else if RingOperator::zero() < coef && j > 0 { 
                coefficient_as_string = vec![String::from("+"), coef.clone().to_string()].concat(); 
            }
            // NOTE: in any other case, we simply call `to_string()`
            else { 
                coefficient_as_string = coef.to_string(); 
            }
            // push the current term to the linear combination to return
            let term = vec![coefficient_as_string, simplex_as_string].concat(); 
            linear_combination.push_str(&term);
        }
        linear_combination
    }
}

